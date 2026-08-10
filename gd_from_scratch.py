"""
경사하강법 직접 구현 — scikit-learn 호환 추정기
================================================
노트북 data24 / data25에서 만든 경사하강법을 재사용 가능한 형태로 정리한 것.

sklearn의 BaseEstimator를 상속했으므로 아래가 전부 그대로 동작한다.
    - Pipeline / ColumnTransformer 안에 넣기
    - cross_val_score / cross_validate 로 교차검증
    - GridSearchCV / RandomizedSearchCV 로 하이퍼파라미터 탐색
    - joblib.dump 로 저장

사용 예
-------
    from gd_from_scratch import GDRegressor, plot_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), GDRegressor(lr=0.1, epochs=1500))
    pipe.fit(X_train, y_train)
    plot_loss(pipe.named_steps['gdregressor'])

주의: 경사하강법은 피처 스케일이 다르면 수렴하지 않는다.
      반드시 StandardScaler를 앞에 붙일 것.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ['GDRegressor', 'GDClassifier', 'plot_loss']


class _GDBase(BaseEstimator):
    """GDRegressor / GDClassifier가 공유하는 학습 루프."""

    def __init__(self, lr=0.05, epochs=1000, batch_size=None, l1=0.0, l2=0.0,
                 shuffle=True, tol=None, random_state=None, verbose=0):
        self.lr = lr                    # 학습률 α — 한 걸음의 보폭
        self.epochs = epochs            # 전체 데이터를 몇 번 반복할지
        self.batch_size = batch_size    # None=배치GD, 1=SGD, n=미니배치GD
        self.l1 = l1                    # L1(Lasso) 규제 강도 — 계수를 0으로
        self.l2 = l2                    # L2(Ridge) 규제 강도 — 계수를 축소
        self.shuffle = shuffle          # 에포크마다 데이터 순서 섞기
        self.tol = tol                  # 손실 개선폭이 이보다 작으면 조기 종료
        self.random_state = random_state
        self.verbose = verbose

    # ---- 하위 클래스가 구현 ----
    def _activation(self, z):
        raise NotImplementedError

    def _data_loss(self, X, y, W, b):
        raise NotImplementedError

    def _penalty(self, W):
        p = 0.0
        if self.l2:
            p += self.l2 * float((W ** 2).sum())
        if self.l1:
            p += self.l1 * float(np.abs(W).sum())
        return p

    def _fit_gd(self, X, y):
        """
        핵심 루프. 회귀(MSE)와 분류(로그손실)의 기울기가
        똑같이  -(1/N)·Xᵀ(y - ŷ)  꼴로 정리되기 때문에 한 함수로 처리된다.
        (회귀는 계수 2가 붙고, 분류는 시그모이드 미분이 상쇄되어 사라진다)
        """
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        W = np.zeros((p, 1))
        b = np.zeros((1, 1))
        bs = n if self.batch_size is None else min(int(self.batch_size), n)

        self.loss_history_ = []
        prev = np.inf

        with np.errstate(over='ignore', invalid='ignore'):
            for ep in range(self.epochs):
                idx = rng.permutation(n) if self.shuffle else np.arange(n)

                for s in range(0, n, bs):
                    j = idx[s:s + bs]
                    Xb, yb = X[j], y[j]
                    N = len(j)

                    # 1) 순전파 — 현재 파라미터로 예측
                    resid = yb - self._activation(Xb @ W + b)     # (N, 1)

                    # 2) 기울기 — 손실이 커지는 방향
                    gW = -(self._grad_scale / N) * (Xb.T @ resid)
                    gb = -(self._grad_scale / N) * resid.sum(keepdims=True)

                    # 3) 규제항의 기울기 (편향 b에는 걸지 않는다)
                    if self.l2:
                        gW += 2 * self.l2 * W
                    if self.l1:
                        gW += self.l1 * np.sign(W)

                    # 4) 하강 — 기울기의 '반대' 방향으로 한 걸음
                    W -= self.lr * gW
                    b -= self.lr * gb

                loss = self._data_loss(X, y, W, b) + self._penalty(W)
                self.loss_history_.append(loss)

                if not np.isfinite(loss):
                    raise FloatingPointError(
                        f'발산했습니다 (epoch {ep}). lr={self.lr}이 너무 큽니다. '
                        f'10분의 1로 줄이거나 StandardScaler를 앞에 붙이세요.')

                if self.verbose and ep % max(1, self.epochs // 10) == 0:
                    print(f'epoch {ep:5d}  loss {loss:.6f}')

                if self.tol is not None and abs(prev - loss) < self.tol:
                    break
                prev = loss

        self.coef_ = W.ravel()
        self.intercept_ = float(b[0, 0])
        self.n_iter_ = len(self.loss_history_)
        self.n_features_in_ = p
        return self


class GDRegressor(RegressorMixin, _GDBase):
    # 믹스인이 반드시 앞에 와야 한다. 뒤에 두면 MRO상 BaseEstimator의
    # __sklearn_tags__가 이겨서 sklearn이 회귀기로 인식하지 못하고,
    # cross_val_score의 scoring이 조용히 nan을 반환한다.
    """
    경사하강법 선형 회귀.   ŷ = Xw + b,   손실 = MSE + 규제

    l2 > 0 이면 Ridge, l1 > 0 이면 Lasso와 같은 역할을 한다.
    (sklearn Ridge(alpha=a)와 맞추려면 l2 = a / n_samples)
    """
    _grad_scale = 2.0        # MSE 미분 계수: dJ/dW = -(2/N)·Xᵀ(y-ŷ)

    def _activation(self, z):
        return z             # 항등 — 선형 회귀

    def _data_loss(self, X, y, W, b):
        return float(((y - (X @ W + b)) ** 2).mean())

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        return self._fit_gd(X, y.reshape(-1, 1))

    def predict(self, X):
        check_is_fitted(self)
        return check_array(X) @ self.coef_ + self.intercept_


class GDClassifier(ClassifierMixin, _GDBase):
    """
    경사하강법 로지스틱 회귀 (이진 분류).
    ŷ = σ(Xw + b),   손실 = 로그손실(binary cross-entropy) + 규제

    시그모이드의 미분이 로그손실의 미분과 상쇄되어,
    기울기가 회귀와 똑같은 -(1/N)·Xᵀ(y-ŷ) 꼴로 떨어진다.
    """
    _grad_scale = 1.0        # 로그손실 미분: dJ/dW = -(1/N)·Xᵀ(y-ŷ)

    @staticmethod
    def _sigmoid(z):
        # z가 크게 음수일 때 exp 오버플로를 피하는 안정적인 형태
        out = np.empty_like(z, dtype=float)
        pos, neg = z >= 0, z < 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def _activation(self, z):
        return self._sigmoid(z)

    def _data_loss(self, X, y, W, b):
        p = np.clip(self._sigmoid(X @ W + b), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError(f'이진 분류만 지원합니다. 클래스 {len(self.classes_)}개가 들어왔습니다.')
        y01 = (y == self.classes_[1]).astype(float).reshape(-1, 1)
        return self._fit_gd(X, y01)

    def decision_function(self, X):
        check_is_fitted(self)
        return check_array(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p1 = self._sigmoid(self.decision_function(X).reshape(-1, 1)).ravel()
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]


def plot_loss(model, ax=None, logy=True):
    """
    학습 곡선을 그린다. 수렴 여부를 눈으로 확인하는 가장 빠른 방법.

        내려가다 평평해짐  -> 정상 수렴
        계속 가파르게 내려감 -> epochs를 늘릴 것
        들쭉날쭉하거나 치솟음 -> lr을 줄일 것
    """
    import matplotlib.pyplot as plt
    check_is_fitted(model)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.loss_history_, lw=1.5)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (log scale)' if logy else 'loss')
    ax.set_title(f'{type(model).__name__}  lr={model.lr}  '
                 f'final={model.loss_history_[-1]:.6g}')
    ax.grid(alpha=.3)
    return ax
