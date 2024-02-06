### A Pluto.jl notebook ###

# ╔═╡ a084cb98-94ed-4d89-8d3e-bdfb89c1c92b
md"""
# POLI 175 - Lecture 07

## Regression (final)"""

# ╔═╡ e2420ffc-fb77-4a0e-a6b7-abe762f6d852
md"""
## Regression and Classification

Loading packages:"""

# ╔═╡ 3bee9e20-b96a-46ad-a0f5-76f5105b00a5
begin
    # If needed
    using Pkg
    Pkg.add("Lowess")
    Pkg.add("Gadfly")
    Pkg.add("RegressionTables")
    Pkg.add("CovarianceMatrices")
    Pkg.add("Econometrics")
    Pkg.add("LinearAlgebra")
    Pkg.add("MixedModelsExtras")
    Pkg.update()
end

# ╔═╡ 758cf533-d6f8-44f3-9b68-ea233e93d88c
md"""
## Regression and Classification

Loading packages:"""

# ╔═╡ b88e91b6-7b63-43f4-8b17-24d896aea596
begin
    ## Loading the data
    using CSV, DataFrames, Plots, GLM, StatsBase, Random
    using LaTeXStrings, StatsPlots, Lowess, Gadfly, RegressionTables
    using CovarianceMatrices, Econometrics, LinearAlgebra, MixedModelsExtras
    
    # Auxiliar function
    function pairplot(df)
        _, cols = size(df)
        plots = []
        for row = 1:cols, col = 1:cols
            push!(
                plots,
                scatter(
                    df[:, row],
                    df[:, col],
                    xtickfont = font(4),
                    ytickfont = font(4),
                    legend = false,
                ),
            )
        end
        Plots.plot(plots..., layout = (cols, cols))
    end
end

# ╔═╡ 6a8ed16b-0820-40cc-95e1-85f3d1f340a0
md"""
## Regression and Classification

Loading data:"""

# ╔═╡ 85f7942b-2c17-42ea-8ed1-ef71a1a0b430
begin
    # URL of the prestige dataset
    urldat = "https://raw.githubusercontent.com/umbertomig/POLI175julia/main/data/Duncan.csv"
    
    # Load the CSV file
    dat = CSV.read(download(urldat), DataFrame)
    
    # First few obs
    first(dat, 3)
end

# ╔═╡ d7045f8d-4f49-4d90-b323-4c87a3187fc7
md"""
## Regression

There are many packages in Julia to run Regression and Classification models. We are going to use two:

- [`GLM`](https://github.com/JuliaStats/GLM.jl) and its family (https://juliastats.org/)
- [`MLJ`](https://alan-turing-institute.github.io/MLJ.jl/dev/)

We have been using the [GLM](https://github.com/JuliaStats/GLM.jl). Next lecture we will center on the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)."""

# ╔═╡ 41d4b496-b49e-4b7d-92ec-c47fe3dab752
md"""
## Regression

### Questions

Quick reminder of a few relevant questions on the prestige of professions dataset:

Questions we answered:

- Is there a relationship between `prestige` and `income`?
- How strong is the relationship between `prestige` and `income`?
- Which variables are associated with `prestige`?
- How can we accurately predict the prestige of professions not studied in this survey?

Questions that we are still to answer:

- Is the relationship linear?
- Is there a synergy among predictors?"""

# ╔═╡ 13b4544d-26c9-4467-9463-bfb849f263d1
md"""
## Regression

### Simplest Model"""

# ╔═╡ cd118105-6bcb-443c-8119-43625cdafa72
mod1 = lm(@formula(prestige ~ income), dat)

# ╔═╡ 4d28ac01-6c33-40db-b893-a9f49b3beb58
md"""
## Diagnostics

Several plots can help us diagnose the quality of our model.

**Warning**: Find and analyzing these violations is **more of an art**.

A careful analysis is frequent enough to ensure you have a `good` model."""

# ╔═╡ eca5ef9a-0d7f-487c-9c1e-f57087de1763
md"""
## Diagnostics

### Non-linearity

When the relationship is non-linear, you could have done better using a different (more flexible) functional form.

The plot to detect this is residual in the y-axis against the fitted values in the x-axis:

![reg](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/fig5.png?raw=true)

**Plot**: Fitted Values x Raw Residuals

- Good: You should find no patterns.

- Bad: A discernible pattern tells you that you could have done better with a more flexible model."""

# ╔═╡ 64405aae-191b-421d-8871-6c0471368e8e
md"""
## Diagnostics

### Non-linearity

But how to find the fitted values and the residuals in the GLM package?"""

# ╔═╡ 90287425-8acb-426c-9bac-924f6099d6c1
begin
    # Residuals
    p1 = histogram(residuals(mod1))
    title!("Residuals")
    p2 = histogram(fitted(mod1))
    title!("Fitted Values")
    Plots.plot(p1, p2, layout = (1, 2))
end

# ╔═╡ 3fef7f2c-6150-4ea2-a3c6-1788035c937f
md"""
## Diagnostics

### Non-linearity

For the `prestige` x `income` relationship:"""

# ╔═╡ 54cb161a-bbde-4018-97ca-7a817ae29d04
# Residual x fitted values (linearity + heteroscedasticity)
Plots.scatter(
    fitted(mod1), residuals(mod1),
    xlabel = "Fitted", ylabel = "Rediduals",
    series_annotations = text.(dat.profession, :left, :bottom, 8),
    legend = false
)

# ╔═╡ 2a8a81f5-2388-4285-a84b-e9ab0abc5953
md"""
## Diagnostics

### Non-linearity

Hint: Look at the smoothing trend line (the `lowess`). You should see no discernible trend."""

# ╔═╡ 2f70f05c-a58e-4d2b-a4dc-2ea15ac732b3
# Residual x fitted values (linearity + heteroscedasticity)
Gadfly.plot(x=fitted(mod1), y=residuals(mod1), 
    Geom.point, Geom.smooth, Guide.xlabel("Fitted"), 
    Guide.ylabel("Residuals"))

# ╔═╡ efbaca31-815c-4354-9048-95c2725e6881
md"""
## Diagnostics

### Non-linearity

Let's `cook` a non-linear relation:

$$ Y = 2 + X + 2 X^2 + \varepsilon $$"""

# ╔═╡ 83adda0e-6fea-43c4-a392-0e2b13e3c278
begin
    ## Cooking
    Random.seed!(4321)
    cooked_data = DataFrame(x = randn(100))
    cooked_data.y = 2 .+ 1 .* cooked_data.x .+ 2 .* (cooked_data.x .^ 2) .+ randn(100)
    
    ## Fitting (wrong)
    mod2 = lm(@formula(y ~ x), cooked_data)
end

# ╔═╡ 7ab14412-fd85-49a6-b2fb-60f8b04ad3c9
md"""
## Diagnostics

### Non-linearity

The smoothing trend line (the `lowess`) show a discernible trend."""

# ╔═╡ 900d7267-c186-429d-b6b1-30d050eb13a1
# Residual x fitted values (linearity + heteroscedasticity)
Gadfly.plot(x=fitted(mod2), y=residuals(mod2), 
    Geom.point, Geom.smooth, Guide.xlabel("Fitted"), 
    Guide.ylabel("Residuals"))

# ╔═╡ 1432d6c9-3fb1-4190-9890-84960eac8e40
md"""
## Diagnostics

### Non-linearity

Let us fit the *right* model now: $$ Y = 2 + X + 2 X^2 + \varepsilon $$"""

# ╔═╡ 4d201541-560f-4d7f-8b11-06e99381d2fc
mod3 = lm(@formula(y ~ x + exp(x)), cooked_data)

# ╔═╡ 4c28352c-0d06-415a-87e4-9a39278cf644
md"""
## Diagnostics

### Non-linearity

And the residuals versus fitted values look better when fitting the correctly specified model:"""

# ╔═╡ 89a9ac92-e5a3-4e27-8003-d3c3fd443e65
# Residual x fitted values (linearity + heteroscedasticity)
Gadfly.plot(x=fitted(mod3), y=residuals(mod3), 
    Geom.point, Geom.smooth, Guide.xlabel("Fitted"), 
    Guide.ylabel("Residuals"))

# ╔═╡ eafd4438-7e3f-42d7-ad35-8e3434ae38de
md"""
## Diagnostics

### Non-linearity

There exist a test called [Ramsey RESET test](https://en.wikipedia.org/wiki/Ramsey_RESET_test).

I strongly suggest you not to use these, since it usually does not identify better relationships than polynomial, while non-linearity can be something extremely complex.

We will learn how to deal with this in a few lectures."""

# ╔═╡ 5adfaf21-fd94-4820-903a-6f3d778ddfd8
md"""
<div class=\"cite2c-biblio\"></div>## Diagnostics

### Heteroscedasticity

It is fancy wording to say that the variance in error is not constant.

It usually means that you are better at fitting some range of the predictors than others.

![reg](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/fig7.png?raw=true)

**Plot:** Fitted Values x Residuals

- Bad (left-hand side): A funnel-shaped figure tells you that you may have heteroscedasticity. It invalidates simple standard errors assumptions.

- Good (right-hand side): You should find no patterns."""

# ╔═╡ 97f2bb67-f84c-408a-b6ae-384ec183089a
md"""
## Diagnostics

### Heteroscedasticity

For the `prestige` x `income` relationship:"""

# ╔═╡ 2a2dc498-370c-4f56-8b51-b4b9d45d650f
# Residual x fitted values (linearity + heteroscedasticity)
Gadfly.plot(
    x = fitted(mod1), y = residuals(mod1),
    label = dat.profession,
    Geom.point, 
    Geom.smooth,
    Geom.label,
    Guide.xlabel("Fitted"), 
    Guide.ylabel("Residuals"),
    Guide.title("Fitted versus raw residual plot"))

# ╔═╡ 6fbf6846-ae90-4dad-bb5d-b03964152413
md"""
## Diagnostics

### Heteroskedasticity

Let us `cook` a heteroskedastic model:

$$ Y = 2 + 3 X + \tilde{\varepsilon} $$

Where $\text{Cov}[\tilde{\varepsilon}] \ \neq \ \sigma^2I$.

In this particular case, let us make the variance of the residuals to look like a football:"""

# ╔═╡ f765f937-4393-4b39-a1f1-8ce451c53e15
begin
    ## Cooking (Het-error term)
    Random.seed!(4321)
    cooked_data = DataFrame(x = randn(100))
    cooked_data.y = 2 .+ 3 .* cooked_data.x .+ (maximum(cooked_data.x .^ 2 .+ 1) .- ((cooked_data.x).^2)) .* randn(100);
end

# ╔═╡ c9747753-b53a-4271-b302-aca24bc3455a
## Fitting
mod4 = lm(@formula(y ~ x), cooked_data)

# ╔═╡ de92528c-5982-47db-bfed-4eaaaf717801
md"""
## Diagnostics

### Heteroscedasticity

For the cooked data:"""

# ╔═╡ 01436eb0-bf68-4f29-a5ac-a83967375d77
# Residual x fitted values (linearity + heteroscedasticity)
Gadfly.plot(x=fitted(mod4), y=residuals(mod4), 
    Geom.point, Geom.smooth, Guide.xlabel("Fitted"), 
    Guide.ylabel("Residuals"))

# ╔═╡ 85034ac3-cdc0-4a45-91d8-b04643b97614
md"""
## Diagnostics

### Heteroscedasticity

Our standard errors are wrong. So are our:

1. Confidence Intervals
1. P-values
1. T-stats

In these cases, we need **robust standard errors**:"""

# ╔═╡ b79e0f28-055b-446f-80a3-aefcc23a5890
# Standard Errors for the heteroskedastic model
CovarianceMatrices.stderror(mod4)

# ╔═╡ bf63f3a1-c720-4c57-9cc9-dc47d1a29918
# Corrected Standard Errors for the heteroskedastic model
# Note: several types of corrections...
CovarianceMatrices.stderror(CovarianceMatrices.HC1(), mod4)

# ╔═╡ d52941d6-bbc2-45f8-9d62-73bb106154b3
md"""
### Outliers

- Outliers are values very far away from most values predicted by the model.

- Sometimes, it is correct, but frequently it may tell you that you made a mistake in collecting the data!

![reg](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/fig8.png?raw=true)

**Plot**: Fitted x Studentized residuals (right-hand side plot)

- Best: You should find no extreme values in the plot.

- Bad: An extreme value can affect your RSE, $R^2$, and mess up with p-values."""

# ╔═╡ fd27a1a0-b83e-4107-9a1e-6cdd860767e7
md"""
## Diagnostics

### Outliers

A few important measurements (all vectors, computed for each data point):

[**Studentized residual**](https://en.wikipedia.org/wiki/Studentized_residual): Residual weighted by the leverage of the point. Studentized because it follows the [Student's t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution).

[**Leverage**](https://en.wikipedia.org/wiki/Leverage_(statistics)): It represents how far away a given *target* value is compared to other *target* values $\bigg(\dfrac{\partial \hat{y}_i}{\partial y_i}\bigg)$.

[**Cook's Distance**](https://en.wikipedia.org/wiki/Cook%27s_distance): It measures how much the regression model changes when we remove a given observation."""

# ╔═╡ c6933bcc-8cad-470c-a4ff-077a0874a356
#= We need to compute the studentized residual
   (https://en.wikipedia.org/wiki/Studentized_residual)
   I wrote this function to facilitate the work
=#
function lm_measures(lmod)
    X = modelmatrix(lmod)
    RSS = sum((residuals(lmod)).^2)
    sigma_hat = sqrt(RSS/dof_residual(lmod))
    leverage = diag(X * inv(transpose(X) * X) * transpose(X))
    studentized_resid = residuals(lmod) ./ (sigma_hat .* sqrt.(1 .- leverage))
    return leverage, studentized_resid, cooksdistance(lmod)
end

# ╔═╡ 37ed1465-3850-4c17-8186-7dfeef94a407
md"""
## Diagnostics

### Outliers"""

# ╔═╡ a5f3e1aa-8055-47ac-b911-7032486f14fb
Gadfly.plot(
    x=fitted(mod1), y=lm_measures(mod1)[2], 
    Geom.point, label = dat.profession, 
    Guide.xlabel("Fitted"),
    Guide.ylabel("Studentized Residuals"), 
    Geom.label,
    yintercept = [-2.0, 2.0], 
    Geom.hline()
)

# ╔═╡ e26d05cc-1718-4298-b0dc-76622c179367
md"""
## Diagnostics

### Outliers"""

# ╔═╡ ae518aa4-49a1-4cf3-965e-78d565347b28
show(dat, allrows = true, allcols = true)

# ╔═╡ a5846e63-90d6-4d4f-98c8-fa071558e5b5
md"""
## Diagnostics

### Outliers"""

# ╔═╡ 82840e2a-8a79-4e08-9667-3bb59b2dd3a1
mod1

# ╔═╡ f1ce2bb6-0904-4d86-b31b-ef51e1a5491f
mod2 = lm(@formula(prestige ~ income), 
    dat[(dat.profession .!= "minister"), :])


# ╔═╡ 58aee0ad-840a-4db3-b1db-498153147d45
mod3 = lm(@formula(prestige ~ income), 
    dat[(dat.profession .!= "conductor"), :])

# ╔═╡ 7fe6f00c-f7cc-4e29-bea4-e34dabfaedfb
mod4 = lm(@formula(prestige ~ income), 
    dat[(dat.profession .!= "minister") .& (dat.profession .!= "conductor"), :])

# ╔═╡ 8d95863d-35c6-4d62-bd99-0d80a03af597
md"""
## Diagnostics

### High Leverage

Having a very unusual value, that could potentially tilt the regression line towards it

***If high leverage and outlier, bad combination!***

![reg](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/fig9.png?raw=true)

**Plot**: Leverage x Studentizided residuals

- Best: You should find no extreme values in the plot.

- Bad: An extreme value can affect your fit."""

# ╔═╡ 2550514d-a1c0-4727-9107-33e6d2e59b84
Gadfly.plot(
    x=lm_measures(mod1)[1], y=lm_measures(mod1)[2], 
    Geom.point, label = dat.profession, 
    Guide.xlabel("Leverage"),
    Guide.ylabel("Studentized Residuals"), 
    Geom.label,
    yintercept = [-2.0, 2.0], 
    Geom.hline()
)

# ╔═╡ 98eaac1e-731b-4601-b5b8-dc8313c1f362
md"""
## Diagnostics

### Cook's Distance

Measures whether removing a given observation tilts the regression coefficients.

**Plot**: Cook's Distance

- Best: You should find no values in the farther diagonal

- Bad: Extreme value in the farther diagonal represents data that is highly influential (high leverage) and high changes in coefficients (high Cook's D)"""

# ╔═╡ e674bb25-4fa9-443f-80e9-22c384ad16bc
Gadfly.plot(
    y = dat.profession, x = lm_measures(mod1)[3], 
    Geom.point
)

# ╔═╡ 847f7c4b-bccd-44de-807a-495ee6c573f6
md"""
## Diagnostics

When we run multiple linear regression, we add *multicollinearity* to the diagnostics we have seen so far.

### Multicollinearity

Multicollinearity is when your predictors are highly correlated. In extreme cases, it messes up with the standard errors in our model (problems with inverse matrix)."""

# ╔═╡ 7a34f51f-1d78-4ad1-9614-d9f9a197a91e
begin
    ## Pairplot to check
    println(first(dat, 1))
    pairplot(dat)
end

# ╔═╡ 9e8c9324-f5ce-4eaa-a489-ce1e91225ad5
md"""
## Diagnostics

### Multicollinearity

One measure of multicollinearity is the [*Variance Inflation Factor*](https://en.wikipedia.org/wiki/Variance_inflation_factor).

How much the multicollinearity is messing up with the estimates.
    
To compute, it is fairly easy. As a rule-of-thumb, we would like to see values lower than 5.

***It is rarely a problem, though... Especially with large datasets.***"""

# ╔═╡ 9173ed47-aaef-441f-ab49-8d44c9ea88d4
begin
    modfull = lm(@formula(prestige ~ income + education + type), dat)
    println(modfull)
    MixedModelsExtras.vif(modfull)
end

# ╔═╡ 4eabce72-b06c-41fc-8d86-585a5cd8a86c
md"""
## Diagnostics

Rules to diagnostics:

1. Always assume heteroskedasticity (use robust standard errors)
1. Check for outliers
1. Drop a few observations and rerun the regression. Do that with most (all) observations.
1. Graph your residuals.
1. If you have extreme values, make sure your results remain valid after dropping them.

In a nutshell, pay attention to what you are doing!"""

# ╔═╡ 85090001-4e34-452f-b90d-9b88565f695b
md"""
# Model Selection"""

# ╔═╡ e29a929e-1ec8-4cd5-87a5-3c470e504646
md"""
## Linear Model Selection

- We usually have a large set of predictors that could be used.
    + Which predictors to pick becomes a task.

- If we are trying to interpret things and learn from the data, then which predictors are correlated with the outcome is informative:
    + Again, picking predictors becomes a task.
    
- In this and the following lecture, we will learn how to do that systematically. """

# ╔═╡ c6b39f32-7be8-4230-81ca-25969e0d86d1
md"""
## Linear Model Selection

### Subset Selection

- In here, we are going to consider techniques to select a subset of predictors based on a performance metric."""

# ╔═╡ f3d669a5-f0ee-4411-a158-3d19cc56345b
md"""
## Linear Model Selection

### Subset Selection

#### Best Subset Selection

**Algorithm:**

1. Let $M_0$ denote the null model, which contains no predictors. This model predicts the sample mean for each observation.

2. For $k = \{1, 2, \cdots, p\}$:
    1. Fit all $p \choose k$ models with exactly $k$ predictors.
    2. Pick the *best* among these models and call it your $M_k$

3. Select a single best model from among $M_0, \cdots, M_p$ using cross-validated prediction error, $C_p$, AIC, BIC, or adjusted $R^2$."""

# ╔═╡ 97cce491-9173-4951-81ee-6981a33fb513
md"""
## Linear Model Selection

### Subset Selection

#### Best Subset Selection

![img ms1](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/ms1.png?raw=true)"""

# ╔═╡ 7f01125d-15b2-438a-a6aa-aaa176812411
md"""
## Linear Model Selection
### Subset Selection

#### Best Subset Selection

Notes:

1. You can do this with Logistic Regression: change RSS with [*deviance*](https://en.wikipedia.org/wiki/Deviance_(statistics)).
    + In this case, our friend $- 2\ln({\hat {L}})$ does very well!

2. Best Selection is excellent but fits around $2^p$ models.
    + $p = 10$ means around 1000 estimates."""

# ╔═╡ 13ec0424-b3bf-4285-a614-c1cc323d3f8c
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Forward Selection

**Algorithm**:

1. Let $M_0$ denote the null model, which contains no predictors.

2. For $k = \{0, 1, 2, \cdots, p-1\}$:
    1. Consider all $p - k$ models that augments $M_k$ by one predictor.
    2. Pick the *best* among these $p-k$ models, and call it your $M_{k+1}$.

3. Select a single best model from among $M_0, \cdots, M_p$ using cross-validated prediction error, $C_p$, AIC, BIC, or adjusted $R^2$."""

# ╔═╡ 30d54e36-654c-4b03-a766-72fbc84b123c
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Forward Selection

- Much more efficient:
    + It fits a total of $1 + \dfrac{p(p+1)}{2}$ models.
    + If $p = 20$, the Best Selection would fit 1,048,576
    + If $p = 20$, the Forward Step Selection would fit 211 models."""

# ╔═╡ a54772b1-c0fd-42c4-a9b1-3927e6deb676
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Forward Selection

- The catch: It is not guaranteed that it is going to find the *best subset* model.

- Example: Let $p = 3$.
    + Suppose that the best model involves $v2$ and $v3$.
    + But suppose that within the models with only one variable, $v1$ would do better.
    + Then, Forward Step Selection would never pick this model!"""

# ╔═╡ ce25d7b9-7209-4729-bb89-617785c92692
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Forward Selection

![img ms2](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/ms2.png?raw=true)"""

# ╔═╡ 90df83fd-0065-4fa5-a885-225229af98c4
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Backward Selection

**Algorithm**:

1. Let $M_p$ denote the *full* model, which contains all $p$ predictors.

2. For $k = \{p, p-1, p-2, \cdots, 1\}$:
    1. Consider all $k$ models that contail all but one predictor in $M_k$, for a total of $k-1$ predictors.
    2. Pick the *best* among these $k$ models, and call it your $M_{k-1}$.

3. Select a single best model from among $M_0, \cdots, M_p$ using cross-validated prediction error, $C_p$, AIC, BIC, or adjusted $R^2$."""

# ╔═╡ c6a1bd9c-4abb-4cb2-ba1a-08194a8477b4
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Backward Selection

- Computational Efficiency:
    + It fits a total of $1 + \dfrac{p(p+1)}{2}$ models.
    + Same efficiency as the Forward Selection.

- Catch:
    + Same catch as the Forward Selection: It does not guarantee the pick of the best model."""

# ╔═╡ 712a94ec-bdba-43c6-8246-24513d200846
md"""
## Linear Model Selection

### Subset Selection

#### Stepwise Selection: Hybrid Approaches

- Combinations of *Forward* and *Backwards* that intend to mimic the *Best Selection*.

- Many available.

- But the trade-offs are clear: 
    + Computational efficiency
    + Likelihood of picking the best model"""

# ╔═╡ f07cc006-a583-4d4c-b280-7b9d3eca6239
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- For each of the $M_k$ models, this is it:
    + RSS: Residual Sum of Squares: We want it to be the lowest possible.
    + $R^2$: We want it to be the highest possible.
    + And for Logistic or other GLM Regressions, *deviance*."""

# ╔═╡ bf75eb38-99d8-4071-9877-6bae9038ad52
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- Catches: 
    1. RSS and $R^2$ always improve with more variables.
    2. We want to look at the *testing set goodness-of-fit*, not the *training sets goodness-of-fit*!

- And that is why RSS and $R^2$ are not used in Step 3:
    - We need something that eventually gets worse the more variables we throw in."""

# ╔═╡ 17c4ee1e-7f54-4e96-b52c-a1f80920b9ec
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

**Important:**

- Training Set MSE generally underestimates the testing set MSE.

$$ \text{MSE} \ = \ \dfrac{RSS}{n} $$

- But before, we could not split data into *training* and *testing*. 
    + This is a more recent feature, thanks to our increased computational power.

- Here are a few stats that we can fit in the training set."""

# ╔═╡ efdf479d-ca30-45aa-86bd-a25a04c37e6b
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- [$C_P$](https://en.wikipedia.org/wiki/Mallows%27s_Cp) in a model containing:
    - $d$ predictors.
    - $n$ observations.
    - $\widehat{\sigma}^2$ the variance of the error in the full model with all predictors.

$$ C_p \ = \ \dfrac{1}{n}(RSS + 2\times d \times \widehat{\sigma}^2) $$

- The smaller, the better."""

# ╔═╡ 378f5ac1-3ce2-4d42-880e-b90ec1f5b63f
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) or $C_p$: AIC is a goodness-of-fit parameters that goes lower when you are improving the model
    + But for every variable you add, it penalizes it.
    + If by adding more variables, it goes up, then your model is getting more complex without adding much.
    
$$ \text{AIC} \ = \ 2d - 2\log({\hat {L}}) $$"""

# ╔═╡ 94d6e978-1f9f-4102-8891-b1e0822d7a52
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- The maximum likelihood and least squares are the same for models with Gaussian errors.

$$ \text{AIC} \ = \ \dfrac{1}{n} (RSS + 2\times d \times \widehat{\sigma}^2) $$"""

# ╔═╡ 9a3e800e-c914-4880-85e8-adc005bb6263
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) is another goodness-of-fit, but this one penalizes the addition of variables at a higher rate.

$$ \text{BIC} = k\log(n) - 2\log({\widehat {L}}) $$

- Note the difference from the AIC: instead of multiplying by 2, it is multiplying by $\ln(n)$!

- Again, lower values are better."""

# ╔═╡ a541f040-8976-41c9-994b-1943169a2fcc
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- Or if you want the one with the least square errors:

$$ \text{AIC} \ = \ \dfrac{1}{n} (RSS + \log(n) \times d \times \widehat{\sigma}^2)  $$ """

# ╔═╡ d7a169cd-2cb5-4d30-a775-76408aaa15a7
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- [Adjusted $R^2$](https://en.wikipedia.org/wiki/Bayesian_information_criterion): It is a change in the $R^2$ to penalize the addition of regressors.

$$ \overline{R}^{2} = 1-(1-R^{2})\dfrac{n-1}{n-d} \ = \ 1 - \dfrac{\frac{RSS}{n - d - 1}}{\frac{TSS}{n - 1}}$$

- $R^2$ always increase but the $\overline{R}^2$ may increase or decrese.

- **Not like the others:** This one, the higher, the better."""

# ╔═╡ 2bc8c76c-e1c5-4082-b295-dc6820f8bd7b
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

![img ms3](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/ms3.png?raw=true)"""

# ╔═╡ 4dd25a3c-81b4-4a9a-9035-6af78b24aca3
md"""
## Linear Model Selection

### Subset Selection

#### What is Best?

- $C_P$, AIC, and BIC all have a solid theoretical justification.
    + Many of them have something we call Large Sample Properties.
    + Check their Wikipedia of them. They converge to nice, important values.

- $\overline{R}^{2}$ does not."""

# ╔═╡ 501c217e-11b3-45e1-8162-89fc3d7a2f50
md"""
## Linear Model Selection

### Subset Selection

#### What is best? Validation and Cross-Validation

- The main advantage is obvious: You look at testing errors!

- The other main advantage is regarding estimating parameters:
    + $C_P$, AIC, and BIC all have a strong theoretical justification, which is reassuring.
    + But sometimes, one does not know the theory behind an estimate to compute statistics or even standard errors.
    + E.g., which $\widehat{\sigma}^2$ should we pick?
    + Validation and Cross-Validation do great in these cases!"""

# ╔═╡ 4eafb29b-9a6c-40c2-8b5a-adcf7f09594b
md"""
## Linear Model Selection

### Subset Selection

#### What is best? Validation and Cross-Validation

![img ms4](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/ms4.png?raw=true)"""

# ╔═╡ 6dbae7a6-2b3d-488a-bc30-efce3042d13f
md"""
# Questions?"""

# ╔═╡ a8525128-c8e7-403a-b3fc-b04a8c2e96e2
md"""
# See you next class
"""

# ╔═╡ Cell order:
# ╟─a084cb98-94ed-4d89-8d3e-bdfb89c1c92b
# ╟─e2420ffc-fb77-4a0e-a6b7-abe762f6d852
# ╠═3bee9e20-b96a-46ad-a0f5-76f5105b00a5
# ╟─758cf533-d6f8-44f3-9b68-ea233e93d88c
# ╠═b88e91b6-7b63-43f4-8b17-24d896aea596
# ╟─6a8ed16b-0820-40cc-95e1-85f3d1f340a0
# ╠═85f7942b-2c17-42ea-8ed1-ef71a1a0b430
# ╟─d7045f8d-4f49-4d90-b323-4c87a3187fc7
# ╟─41d4b496-b49e-4b7d-92ec-c47fe3dab752
# ╟─13b4544d-26c9-4467-9463-bfb849f263d1
# ╠═cd118105-6bcb-443c-8119-43625cdafa72
# ╟─4d28ac01-6c33-40db-b893-a9f49b3beb58
# ╟─eca5ef9a-0d7f-487c-9c1e-f57087de1763
# ╟─64405aae-191b-421d-8871-6c0471368e8e
# ╠═90287425-8acb-426c-9bac-924f6099d6c1
# ╟─3fef7f2c-6150-4ea2-a3c6-1788035c937f
# ╠═54cb161a-bbde-4018-97ca-7a817ae29d04
# ╟─2a8a81f5-2388-4285-a84b-e9ab0abc5953
# ╠═2f70f05c-a58e-4d2b-a4dc-2ea15ac732b3
# ╟─efbaca31-815c-4354-9048-95c2725e6881
# ╠═83adda0e-6fea-43c4-a392-0e2b13e3c278
# ╟─7ab14412-fd85-49a6-b2fb-60f8b04ad3c9
# ╠═900d7267-c186-429d-b6b1-30d050eb13a1
# ╟─1432d6c9-3fb1-4190-9890-84960eac8e40
# ╠═4d201541-560f-4d7f-8b11-06e99381d2fc
# ╟─4c28352c-0d06-415a-87e4-9a39278cf644
# ╠═89a9ac92-e5a3-4e27-8003-d3c3fd443e65
# ╟─eafd4438-7e3f-42d7-ad35-8e3434ae38de
# ╟─5adfaf21-fd94-4820-903a-6f3d778ddfd8
# ╟─97f2bb67-f84c-408a-b6ae-384ec183089a
# ╠═2a2dc498-370c-4f56-8b51-b4b9d45d650f
# ╟─6fbf6846-ae90-4dad-bb5d-b03964152413
# ╠═f765f937-4393-4b39-a1f1-8ce451c53e15
# ╠═c9747753-b53a-4271-b302-aca24bc3455a
# ╟─de92528c-5982-47db-bfed-4eaaaf717801
# ╠═01436eb0-bf68-4f29-a5ac-a83967375d77
# ╟─85034ac3-cdc0-4a45-91d8-b04643b97614
# ╠═b79e0f28-055b-446f-80a3-aefcc23a5890
# ╠═bf63f3a1-c720-4c57-9cc9-dc47d1a29918
# ╟─d52941d6-bbc2-45f8-9d62-73bb106154b3
# ╟─fd27a1a0-b83e-4107-9a1e-6cdd860767e7
# ╠═c6933bcc-8cad-470c-a4ff-077a0874a356
# ╟─37ed1465-3850-4c17-8186-7dfeef94a407
# ╠═a5f3e1aa-8055-47ac-b911-7032486f14fb
# ╟─e26d05cc-1718-4298-b0dc-76622c179367
# ╠═ae518aa4-49a1-4cf3-965e-78d565347b28
# ╟─a5846e63-90d6-4d4f-98c8-fa071558e5b5
# ╠═82840e2a-8a79-4e08-9667-3bb59b2dd3a1
# ╠═f1ce2bb6-0904-4d86-b31b-ef51e1a5491f
# ╠═58aee0ad-840a-4db3-b1db-498153147d45
# ╠═7fe6f00c-f7cc-4e29-bea4-e34dabfaedfb
# ╟─8d95863d-35c6-4d62-bd99-0d80a03af597
# ╠═2550514d-a1c0-4727-9107-33e6d2e59b84
# ╟─98eaac1e-731b-4601-b5b8-dc8313c1f362
# ╠═e674bb25-4fa9-443f-80e9-22c384ad16bc
# ╟─847f7c4b-bccd-44de-807a-495ee6c573f6
# ╠═7a34f51f-1d78-4ad1-9614-d9f9a197a91e
# ╟─9e8c9324-f5ce-4eaa-a489-ce1e91225ad5
# ╠═9173ed47-aaef-441f-ab49-8d44c9ea88d4
# ╟─4eabce72-b06c-41fc-8d86-585a5cd8a86c
# ╟─85090001-4e34-452f-b90d-9b88565f695b
# ╟─e29a929e-1ec8-4cd5-87a5-3c470e504646
# ╟─c6b39f32-7be8-4230-81ca-25969e0d86d1
# ╟─f3d669a5-f0ee-4411-a158-3d19cc56345b
# ╟─97cce491-9173-4951-81ee-6981a33fb513
# ╟─7f01125d-15b2-438a-a6aa-aaa176812411
# ╟─13ec0424-b3bf-4285-a614-c1cc323d3f8c
# ╟─30d54e36-654c-4b03-a766-72fbc84b123c
# ╟─a54772b1-c0fd-42c4-a9b1-3927e6deb676
# ╟─ce25d7b9-7209-4729-bb89-617785c92692
# ╟─90df83fd-0065-4fa5-a885-225229af98c4
# ╟─c6a1bd9c-4abb-4cb2-ba1a-08194a8477b4
# ╟─712a94ec-bdba-43c6-8246-24513d200846
# ╟─f07cc006-a583-4d4c-b280-7b9d3eca6239
# ╟─bf75eb38-99d8-4071-9877-6bae9038ad52
# ╟─17c4ee1e-7f54-4e96-b52c-a1f80920b9ec
# ╟─efdf479d-ca30-45aa-86bd-a25a04c37e6b
# ╟─378f5ac1-3ce2-4d42-880e-b90ec1f5b63f
# ╟─94d6e978-1f9f-4102-8891-b1e0822d7a52
# ╟─9a3e800e-c914-4880-85e8-adc005bb6263
# ╟─a541f040-8976-41c9-994b-1943169a2fcc
# ╟─d7a169cd-2cb5-4d30-a775-76408aaa15a7
# ╟─2bc8c76c-e1c5-4082-b295-dc6820f8bd7b
# ╟─4dd25a3c-81b4-4a9a-9035-6af78b24aca3
# ╟─501c217e-11b3-45e1-8162-89fc3d7a2f50
# ╟─4eafb29b-9a6c-40c2-8b5a-adcf7f09594b
# ╟─6dbae7a6-2b3d-488a-bc30-efce3042d13f
# ╟─a8525128-c8e7-403a-b3fc-b04a8c2e96e2
