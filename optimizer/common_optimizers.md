# Common Optimizers

This note summarizes widely used gradient-based optimizers and their update rules.

Notation:
- $\theta_t$: parameters at step $t$
- $L(\theta)$: objective (loss) function
- $g_t = \nabla L(\theta_t)$: gradient at step $t$
- $\eta$: learning rate
- $\epsilon$: numerical stability term
- $\odot$: elementwise product
- $g_t^2$: elementwise square of the gradient
- $v_t$: velocity (momentum) term
- $r_t$: accumulated (or EMA) squared gradients
- $m_t$: first moment estimate (EMA of gradients)
- $\hat{m}_t$, $\hat{v}_t$: bias-corrected moment estimates
- $\mu$: momentum coefficient
- $\rho$: RMSProp decay rate
- $\beta_1$, $\beta_2$: Adam moment decay rates
- $\lambda$: decoupled weight decay coefficient

## SGD (Gradient-Based)

Summary: Moves parameters in the negative gradient direction using a fixed or scheduled learning rate.

Update rule:
$$
\theta_{t+1} = \theta_t - \eta g_t
$$

Key hyperparameters:
- $\eta$ (learning rate)

Notes:
- Simple and often strong when combined with learning-rate schedules.
- Sensitive to feature scaling and noisy gradients.

## SGD with Momentum (Momentum-Based)

Summary: Accumulates a velocity vector to smooth gradients and accelerate in consistent directions.

Update rule:
$$
v_{t+1} = \mu v_t + g_t
$$
$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

Key hyperparameters:
- $\eta$ (learning rate)
- $\mu$ (momentum coefficient, typically 0.9)

Notes:
- Helps traverse ravines and reduce oscillation.
- Nesterov momentum is a common variant that looks ahead.

## AdaGrad

Summary: Adapts the learning rate per-parameter based on accumulated squared gradients.

Update rule:
$$
r_{t+1} = r_t + g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_{t+1}} + \epsilon} \odot g_t
$$

Key hyperparameters:
- $\eta$ (learning rate)
- $\epsilon$ (numerical stability)

Notes:
- Works well for sparse features.
- Learning rates can decay too aggressively over time.

## RMSProp

Summary: Uses an exponential moving average of squared gradients to prevent AdaGrad's rapid decay.

Update rule:
$$
r_{t+1} = \rho r_t + (1 - \rho) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_{t+1}} + \epsilon} \odot g_t
$$

Key hyperparameters:
- $\eta$ (learning rate)
- $\rho$ (decay rate, typically 0.9)
- $\epsilon$ (numerical stability)

Notes:
- Common default for RNNs and non-stationary objectives.

## Adam

Summary: Combines momentum on gradients and RMSProp-style adaptive scaling with bias correction.

Update rule:
$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
$$
$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}
$$
$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
$$

Key hyperparameters:
- $\eta$ (learning rate)
- $\beta_1$ (momentum for mean, typically 0.9)
- $\beta_2$ (momentum for variance, typically 0.999)
- $\epsilon$ (numerical stability)

Notes:
- Strong default for many tasks.
- Can generalize worse than SGD in some settings.

## AdamW

Summary: Decouples weight decay from the adaptive update, improving regularization.

Update rule:
$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
$$
$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}
$$
$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} - \eta \lambda \theta_t
$$

Key hyperparameters:
- $\eta$ (learning rate)
- $\beta_1$, $\beta_2$ (Adam moments)
- $\epsilon$ (numerical stability)
- $\lambda$ (decoupled weight decay)

Notes:
- Preferred for training large models when using weight decay.
- Often paired with cosine or warmup schedules.
