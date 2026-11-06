"""Held-out OLS validation and final candidate damage predictions."""
from time import perf_counter

import numpy as np

from .data import CANDIDATES, training_data


def fit_predict():
    """Validate on a fixed split, then refit on all 120 historical records."""
    t=perf_counter()
    X,y=training_data()
    design=np.column_stack([np.ones(len(X)),X])
    order=np.random.default_rng(7).permutation(120)
    train,test=order[:96],order[96:]
    validation_coef=np.linalg.lstsq(design[train],y[train],rcond=None)[0]
    error=y[test]-design[test] @ validation_coef
    metrics=dict(mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                 r2=float(1-sum(error**2)/sum((y[test]-y[test].mean())**2)))
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    predictions=np.column_stack([np.ones(8),CANDIDATES]) @ coef
    ml_seconds=perf_counter()-t
    equation=f'yhat={coef[0]:.6f}'+''.join(f'{b:+.6f}*{name}' for b,name in zip(coef[1:],['d','a','l','r']))

    return dict(predictions=predictions.tolist(), metrics=metrics, equation=equation,
                coefficients=coef.tolist(), ml_seconds=ml_seconds)
