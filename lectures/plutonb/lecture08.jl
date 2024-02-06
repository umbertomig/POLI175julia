### A Pluto.jl notebook ###

# ╔═╡ f749ff71-2d26-4dec-8f6b-7f1353768587
md"""
# POLI 175 - Lecture 08

## Classification"""

# ╔═╡ f99397c2-1b39-458a-9b02-bbd6ca7c6fd5
md"""
## Classification

Loading packages:"""

# ╔═╡ 62a6c9a3-4ceb-4706-8f69-40d79ca105cb
begin
    # If needed
    using Pkg
    Pkg.add("Missings")
    Pkg.add("StatsAPI")
    Pkg.add("FreqTables")
    Pkg.add("EvalMetrics")
    Pkg.update()
end

# ╔═╡ 787f3195-c449-4686-acea-925e90e40973
md"""
## Classification

Loading packages:"""

# ╔═╡ 3df02c92-1d6e-4a85-8f81-ad87eef43d79
begin
    ## Loading the packages
    using CSV, DataFrames, Plots, GLM, StatsBase, Random
    using LaTeXStrings, StatsPlots, Lowess, Gadfly, RegressionTables
    using CovarianceMatrices, Econometrics, LinearAlgebra, MixedModelsExtras
    using Missings, StatsAPI, FreqTables, EvalMetrics
    
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

# ╔═╡ 5e779878-468b-4f81-a532-8e76fb5da797
md"""
# Classification"""

# ╔═╡ 6d2a95eb-0662-4854-b27d-6e8a82c09b39
md"""
## Classification

Linear regression is great! But it assumes we want to predict a continuous target variable.

There are situations when our target is qualitative.

**Examples:**

1. Whether a country default its debt obligations?

1. Whether a person voted Republican, Democrat, Independent, voted for a different party, or did not turnout to vote?

1. What determines the number of FOI requests that a given public office receives every day?

1. Is a country expected to meet, exceed, or not meet the Paris Treaty Nationally Determined Contributions?

All these questions are qualitative in nature."""

# ╔═╡ d0ac4d2c-10f8-46ac-a625-f6b03b9e8ffd
md"""
## Running Example

### [Chile Survey](https://en.wikipedia.org/wiki/Chile)

In 1988, the [Chilean Dictator](https://en.wikipedia.org/wiki/Military_dictatorship_of_Chile) [Augusto Pinochet](https://en.wikipedia.org/wiki/Augusto_Pinochet) conducted a [referendum to whether he should step out](https://en.wikipedia.org/wiki/1988_Chilean_presidential_referendum).

The [FLACSO](https://en.wikipedia.org/wiki/Latin_American_Faculty_of_Social_Sciences) in Chile conducted a surver on 2700 respondents. We are going to build a model to predict their voting intentions.

| **Variable** | **Meaning** |
|:---:|---|
| region | A factor with levels:<br>- `C`, Central; <br>- `M`, Metropolitan Santiago area; <br>- `N`, North; <br>- `S`, South; <br>- `SA`, city of Santiago. |
| population | The population size of respondent's community. |
| sex | A factor with levels: <br>- `F`, female; <br>- `M`, male. |
| age | The respondent's age in years. |
| education | A factor with levels: <br>- `P`, Primary; <br>- `S`, Secondary; <br>- `PS`, Post-secondary. |
| income | The respondent's monthly income, in Pesos. |
| statusquo | A scale of support for the status-quo. |
| vote | A factor with levels: <br>- `A`, will abstain; <br>- `N`, will vote no (against Pinochet);<br>- `U`, is undecided; <br>- `Y`, will vote yes (for Pinochet). |"""

# ╔═╡ 29d8ecb7-4cce-4806-9dab-89f3c6f1dd37
md"""
## Chile Survey"""

# ╔═╡ df57342e-7834-41d1-b691-820316c8d5e9
begin
    ## Loading the data
    chile = CSV.read(
        download("https://raw.githubusercontent.com/umbertomig/POLI175julia/main/data/chilesurvey.csv"), 
        DataFrame,
        missingstring = ["NA"]
    ); dropmissing!(chile)
    chile.voteyes = ifelse.(chile.vote .== "Y", 1, 0)
    first(chile, 3)
end

# ╔═╡ 82243c55-ea4e-48fb-a6d6-73e88a844532
md"""
## Book's Example

Probability of voting `yes` (favor of Pinochet remain in power for eight more years), conditional on how satisfied the person is with things as they are:"""

# ╔═╡ dec2b960-9e14-4f7a-8432-69b27ed43f44
dat = DataFrame(statusquo = collect(range(-3, stop = 3, length = 100)));

# ╔═╡ 2c931a1e-bcbe-4ac1-8af1-cf69c7336b6a
Gadfly.plot(
    x = chile.statusquo, y = chile.voteyes, Stat.y_jitter(range = 0.2), Geom.point,
    Guide.xlabel("Status Quo Preference"), Guide.ylabel("Vote for Pinochet"),
    layer(x = dat.statusquo, y = predict(glm(@formula(voteyes ~ statusquo), chile, Binomial(), LogitLink()), dat), Geom.line, color = [colorant"red"], order = 2),
    layer(x = dat.statusquo, y = predict(lm(@formula(voteyes ~ statusquo), chile), dat), Geom.line, color = [colorant"blue"], order = 3)
)

# ╔═╡ e2b698c5-6ac1-4000-8edf-a9544a35dfed
md"""
## Logistic Regression

[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) belongs to a class of models called [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model) (or GLM for short).

GLM, in a nutshell (and in a proudly lazy definition) is an expansion of Linear Model that assumes:
- A Linear Relationship in part of the model
- But then applies a non-linear transformation to the response variable.

The non-linear transformation is called `link function`. Many link functions around (check [here](https://en.wikipedia.org/wiki/Generalized_linear_model) for various link functions).

The link function will determine which types of models we run.

When the outcome variable is binary, we may use `Logistic` or `Probit` links."""

# ╔═╡ 35830a4e-8594-4352-922e-0c3bde3bd525
md"""
## Logistic Regression

In a regression, we are investigating something along the lines of:

$$ \mathbb{E}[Y | X] \ = \ \beta_0 + \beta_1 X $$

But when the outcome is binary we would like to get:

$$ \mathbb{E}[Y | X] \ = \ \mathbb{P}(Y = 1 | X) $$

And the Logistic link is nothing but:

$$ \mathbb{P}(Y = 1 | X) \ = \ \dfrac{e^{(\beta_0 + \beta_1X)}}{1 + e^{(\beta_0 + \beta_1X)}} \ = \ \dfrac{1}{1 + e^{-(\beta_0 + \beta_1X)}} $$"""

# ╔═╡ e994c99a-491f-4c93-bbe4-87ef873b4e72
md"""
## Logistic Regression

With a bit of manipulation, we get to something called odds ratio:

$$ \dfrac{\mathbb{P}(Y = 1 | X)}{\mathbb{P}(Y = 0 | X)} \ = \ \dfrac{\mathbb{P}(Y = 1 | X)}{1 - \mathbb{P}(Y = 1 | X)} \ = \ e^{(\beta_0 + \beta_1X)} $$

And logging the thing gets rid of the Euler constant:

$$ \log \left( \dfrac{\mathbb{P}(Y = 1 | X)}{1 - \mathbb{P}(Y = 1 | X)}\right) \ = \ \beta_0 + \beta_1X $$

And this is the [Logit Link](https://en.wikipedia.org/wiki/Logistic_regression)."""

# ╔═╡ 7e889ac3-5293-448f-970f-860fd74edcb7
md"""
## Logistic Regression

Little detour to talk about odd ratios:

- Note the odd ratio: $\dfrac{\mathbb{P}(Y = 1 | X)}{1 - \mathbb{P}(Y = 1 | X)}$

- It is a ratio between the chance of $Y = 1$ divided by the chance of $Y = 0$.

- Since probabilities are between zero and one, the ratio is always between $(0, \infty)$.

Example:

- If based on characteristics, two in every ten people vote for Pinochet, $\mathbb{P}(Y = 1 | X = \text{some characs.}) = 0.2$ and the odds ratio is $1/4$.

- If based on other set of characteristics, nine out of ten people vote for Pinochet, $\mathbb{P}(Y = 1 | X = \text{some other characs.}) = 0.9$ and the odds ratio is $9$.

One is the number that *does not change the ratios*.
"""

# ╔═╡ 164b7088-314f-4ad9-adf8-e7d8f1fb937a
md"""
## Logistic Regression

Little other detour to talk about the coefficients:

In linear regression, changes in one unit of $x_i$ changes your target variable in $\beta_i$ units, on average.

In logistic regression, changes in one unit of $x_i$ changes **the log odds** of your target variable in $\beta_i$ units, on average.

- Multiplies the odds by $e^{\beta_i}$! This is **not** a straight line!

- Easy proxy (does not work for interaction terms): 
    + When $\beta_1$ is **positive**, it **increases** the $\mathbb{P}(Y = 1 | X)$
    + When $\beta_1$ is **negative**, it **decreases** the $\mathbb{P}(Y = 1 | X)$
    
Try to compute the partial derivatives on $X$ and you will see the complications!"""

# ╔═╡ 26102ac3-f9d6-412c-bc22-e1b7a99d35d5
md"""
## Logistic Regression

Technical:

1. The estimation is through [maximizing the likelihood function](https://en.wikipedia.org/wiki/Likelihood_function).<br> This is outside the scope of the course, but an interesting topic to learn in an advanced course.<br> <br> 


2. The hypothesis test for the coefficient's significance in here is a Z-test (based on the Normal distribution). <br> Null Hypothesis: $$H_0: \beta_i = 0 \quad \text{ or alternatively, } \quad H_0: e^{\beta_i} = 1$$<br> <br> 


3. Making predictions: Just insert the predicted $\hat{\beta}$s on the equation.
    
$$ \hat{p}(X) \ = \ \dfrac{e^{\hat{\beta}_0 + \hat{\beta}_1 X}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 X}} $$"""

# ╔═╡ 77f96bb5-3f2e-463a-928c-e036971971dc
md"""
## Logistic Regression

Fitting a Linear Regression using Julia GLM package. Note the syntax:

```julia
glm(
    @formula(target ~ feature1 + feature2 + ... + featureK), 
    dataset, 
    distr = Binomial(),
    link = LogitLink()
)
```"""

# ╔═╡ a3ddbf90-5314-458f-badb-fd49ac3bcc20
modlog = glm(@formula(voteyes ~ age), chile, Binomial(), LogitLink())

# ╔═╡ dd37ea65-0307-48fe-bd8a-993ad4024d72
md"""
What is happening here? Let us interpret some of the results."""

# ╔═╡ cd081d49-a5fd-4e62-96a3-58c8bb2bde1d
md"""
## Logistic Regression

Compare with a linear regression:"""

# ╔═╡ af8e439a-8acd-42d7-9751-753ef2b549e3
begin
    # Linear Model
    modlin = lm(@formula(voteyes ~ age), chile)
    regtable(modlin, modlog)
end

# ╔═╡ 45ea2d28-b9da-4b81-aa31-26b927c7b4f0
md"""
What is happening here? Let us interpret some of the results for both models."""

# ╔═╡ 73c1dc27-a4f4-4d07-8491-7504eb6f8d97
md"""
## Logistic Regression

Column 1: [Linear Probability Model](https://en.wikipedia.org/wiki/Linear_probability_model)

Column 2: [Logistic Model](https://en.wikipedia.org/wiki/Logistic_regression)

Column 3: [Probit Model](https://en.wikipedia.org/wiki/Probit_model)

And there are other links that we may use."""

# ╔═╡ 068b8878-e84d-4b78-96b9-834d9a3d2140
begin
    modpro = glm(@formula(voteyes ~ age), chile, Binomial(), ProbitLink())
    regtable(modlin, modlog, modpro)
end

# ╔═╡ db16c15b-da56-4b2f-8d58-449fc93d5147
md"""
## Logistic Regression

We can easily fit multivariate regressions:

Model 1: [Linear Probability Model](https://en.wikipedia.org/wiki/Linear_probability_model)

Model 2: [Logistic Model](https://en.wikipedia.org/wiki/Logistic_regression)

Model 3: [Probit Model](https://en.wikipedia.org/wiki/Probit_model)"""

# ╔═╡ a7456777-308c-4774-afa6-55f66b58f78e
begin
    modlin2 = lm(@formula(voteyes ~ region + log(population) + sex + age + education + log(income)), chile);
    modlog2 = glm(@formula(voteyes ~ region + log(population) + sex + age + education + log(income)), chile, Binomial(), LogitLink());
    modpro2 = glm(@formula(voteyes ~ region + log(population) + sex + age + education + log(income)), chile, Binomial(), ProbitLink());
end

# ╔═╡ 540ba93f-5073-469a-b409-5aa4acb6e665
md"""
## Logistic Regression

Now with a pretty table:"""

# ╔═╡ f89cc1c6-0afe-4179-a3b5-d29567b7901c
regtable(modlin2, modlog2, modpro2; 
    groups = ["Linear", "Logistic", "Probit"],
    number_regressions = false,
    labels = Dict("voteyes" => "Vote for Pinochet", "(Intercept)" => "Constant", "region: M" => "Region = Santiago Metro",
        "region: N" => "Region = North", "region: S" => "Region = South", 
        "region: SA" => "Region = Santiago City", "log(population)" => "Log District Population",
        "sex: M" => "Male", "age" => "Age", "education: PS" => "Post-Secondary Education",
        "education: S" => "Secondary Education", "log(income)" => "Log District Income"
    )
)

# ╔═╡ 99de1ab8-9fc5-4cbf-b600-d0ab99a953ea
md"""
## Logistic Regression

The predicted values are probabilities:"""

# ╔═╡ f9d4399b-b255-4319-ad83-3e24028498b3
Gadfly.plot(
    x = predict(modlog2), 
    Geom.histogram, 
    color = ifelse.(predict(modlog2).>0.5, "Predict Yes", "Predict No"),
    Guide.xlabel("Predicted Probabilities")
)

# ╔═╡ de595011-ae08-403e-aa0a-23341c6b6289
md"""
## Logistic Regression

Let's look at the coefficients. Those are log odds ratio:"""

# ╔═╡ 282e665e-704a-455f-90a6-a3e947b4fbab
begin
    ## Parameters
    println(modlog2)
    println(coef(modlog2))
end

# ╔═╡ 6d3120a0-3a7b-4f29-978f-078713dc3340
md"""
## Logistic Regression

To make them odds ratio, we exponentiate."""

# ╔═╡ c77d51e2-3e5c-4c2e-a994-3855927cf83e
begin
    ## Parameters
    println(modlog2)
    println(exp.(coef(modlog2)))
end

# ╔═╡ 77358e9b-d2e4-4490-af18-ac9f572ca876
md"""
## Logistic Regression

**The interaction term**: 

We found in the previous models that:

1. The higher the district's population, the lower the chance of supporting Pinochet
1. The higher the district's income, the higher the chance of voting for Pinochet

The first result relates with people living in cities preferring democracy; the second results relate to people that are benefiting financially from the status quo preferring to keep things as they are.

But how about a rich person that lives in a city? In the US analogy, thing of billionaire democrat. She supports the democrats even though she may be benefiting from things they are.

Conversely, how about a poor person that lives in a village? It could be that the person is not benefiting from things they are, but she favors Pinochet for conservative issues.

Let us prepare a situation where population stays fixed on one of three values: 25th percentile (small villages), median (median-density districts), 75th percentile (more dense districts); and income varying from zero to one million pesos.

This should give an idea of how population and income jointly affect the probability of voting for Pinochet."""

# ╔═╡ b67db2ce-5d11-4a62-90c4-b42d69618459
begin
    # Auxiliary variables
    chile.pop100k = chile.population / 100000;
    chile.inc1mi = chile.income / 1000000;
end

# ╔═╡ 6c0e975e-f696-41ae-b56d-ca43d26f7d6a
begin
    # Three interactions between a fixed income and a varying population
    newdat_popq1 = DataFrame(
        pop100k = repeat([quantile(chile.pop100k, 0.25)], 100), 
        inc1mi = collect(range(0.0, stop = 1.0, length = 100))
    );
    newdat_popmed = DataFrame(
        pop100k = repeat([median(chile.pop100k)], 100), 
        inc1mi = collect(range(0.0, stop = 1.0, length = 100))
    );
    newdat_popq3 = DataFrame(
        pop100k = repeat([quantile(chile.pop100k, 0.75)], 100), 
        inc1mi = collect(range(0.0, stop = 1.0, length = 100))
    );
end

# ╔═╡ 97bb48e4-26c2-4b7e-904c-fad63d399ae0
md"""
## Logistic Regression

**The interaction term**: 

In a linear regression:"""

# ╔═╡ 5697d8af-5f02-451c-873d-0cbb3241d15c
begin
    ## Parameters
    modlin3 = lm(@formula(voteyes ~ pop100k * inc1mi ), chile)
    regtable(modlin3; 
        number_regressions = false,
        labels = Dict(
            "voteyes" => "Vote for Pinochet", 
            "(Intercept)" => "Constant", 
            "pop100k" => "Population (100k people)",
            "inc1mi" => "Income (1 million pesos)",
            "pop100k & inc1mi" => "Population x Income"
        )
    )
end

# ╔═╡ 3d455dee-090e-4014-9e09-12c3a7ce3672
md"""
## Logistic Regression

**The interaction term**:

And here is how the interaction term affects the results:"""

# ╔═╡ 9af09918-9e7e-4865-8606-28a9e6629608
Gadfly.plot(x = newdat_popq1.inc1mi, y=predict(modlin3, newdat_popq1), Geom.line, color = ["Population (Q1)"],
    layer(x = newdat_popmed.inc1mi, y=predict(modlin3, newdat_popmed), Geom.line, color = ["Population (median)"]),
    layer(x = newdat_popq3.inc1mi, y=predict(modlin3, newdat_popq3), Geom.line, color = ["Population (Q3)"]),
    Guide.xlabel("Income"),
    Guide.ylabel("Chance of Voting for Pinochet"),
    Guide.title("Linear Probability Model") 
)

# ╔═╡ 9d8a601f-48fd-4447-90df-b8ccd7648f21
md"""
Note the ***linearity*** of the interaction effect. Note also the effects of extrapolating."""

# ╔═╡ b064e023-29c4-4832-be0d-988065de253f
md"""
## Logistic Regression

**The interaction term**: 

In the logistic regression, you need to use a plot called [interactive effects plot](). If you are curious, [here is a good theoretical explanation, with Stata examples](https://stats.oarc.ucla.edu/stata/seminars/deciphering-interactions-in-logistic-regression/)."""

# ╔═╡ 12ca9c1a-9f3c-4569-aafe-3dbf2b8e885c
begin
    ## Parameters
    modlog3 = glm(@formula(voteyes ~ pop100k * inc1mi), chile, Binomial(), LogitLink())
    regtable(modlog3; 
        number_regressions = false,
        labels = Dict(
            "voteyes" => "Vote for Pinochet", 
            "(Intercept)" => "Constant", 
            "pop100k" => "Population (100k people)",
            "inc1mi" => "Income (1 million pesos)",
            "pop100k & inc1mi" => "Population x Income"
        )
    )
end

# ╔═╡ c17a10c6-994e-407f-bbaf-ddc18b23e39e
md"""
## Logistic Regression

**The interaction term**:

Note now the ***non-linearity*** of the interaction effect:"""

# ╔═╡ 646d785e-f8ff-439e-87b9-02a0e27f024e
Gadfly.plot(x = newdat_popq1.inc1mi, y=predict(modlog3, newdat_popq1), Geom.line, color = ["Population (Q1)"],
    layer(x = newdat_popmed.inc1mi, y=predict(modlog3, newdat_popmed), Geom.line, color = ["Population (median)"]),
    layer(x = newdat_popq3.inc1mi, y=predict(modlog3, newdat_popq3), Geom.line, color = ["Population (Q3)"]),
    Guide.xlabel("Income"),
    Guide.ylabel("Chance of Voting for Pinochet"),
    Guide.title("Logistic Model") 
)

# ╔═╡ 3d3a253f-5a3c-4f3b-9a3f-9ce900bf8fe1
md"""
## Logistic Regression

In generalized linear models where the link function is not the identity function (i.e., not a linear regression), you need to be careful when interpreting and using interactions.

In linear models, it is straightforward.

In all models, it involves taking a partial derivative of the conditional expectation function that we fit in the training stage.

A catch is: This discussion matters more if you care about interpreting your model. For prediction, it does not matter that much.

***But what matters for prediction?***"""

# ╔═╡ 29ac4033-a530-4b52-a703-95bd40393038
md"""
## Logistic Regression

### Prediction

Suppose that we are an investment firm that will invest in Chile only if democracy wins.

Because the plebiscite did not happened, uncertainty may bring its own gains. Thus, it may be better to invest *before* the results.

All we have is this dataset. So, how much we predict modeling the vote for pinochet using these variables?

Let us model vote for Pinochet using the following variables: `log(income)`, `log(population)`, and `age`."""

# ╔═╡ 50a9393c-4e21-44b8-9a57-93b68774cb41
modlog4 = glm(@formula(voteyes ~ log(population) + log(income) + age), chile, Binomial(), LogitLink())

# ╔═╡ 6fce3d3a-b38c-43e8-887b-9ef190808a8c
md"""
How well is this model doing?"""

# ╔═╡ 36057635-934c-4226-93d7-7232c7bb4b4a
md"""
## Logistic Regression

### Prediction

We have significance, and we may think: pretty well! But like the $R^2$ for linear regressions, we need something that works similar in here.

Let us start checking how people voted:"""

# ╔═╡ 39add574-43cc-4e32-bb64-32f9fb6e92e9
freqtable(chile.voteyes)

# ╔═╡ b3905735-463e-4c3a-8757-9788f04a21b5
proptable(chile.voteyes)

# ╔═╡ 5eb8f5aa-f2f7-480e-bb33-437ff7263efa
md"""
And if you were to guess, what would that guess be? ***This is our benchmark!***"""

# ╔═╡ 86f6d905-2e43-4171-b840-2b262620166d
md"""
## Logistic Regression

### Prediction

Now let us look at our model. 

For simplicity, we will say that if the model gives more than 0.5 chance of favoring Pinochet, the person voted in favor. Otherwise, the person voted against."""

# ╔═╡ b8bba81a-a388-4429-b461-be5aa9c28023
Gadfly.plot(
    x = predict(modlog4), 
    Geom.histogram, 
    color = ifelse.(predict(modlog4).>0.5, "Predict Yes", "Predict No"),
    Guide.xlabel("Predicted Probabilities")
)

# ╔═╡ be85936d-6da3-4fc0-a2d6-857c91096015
md"""
## Logistic Regression

### Prediction

Now let us look at our model. 

For simplicity, we will say that if the model gives more than 0.5 chance of favoring Pinochet, the person voted in favor. Otherwise, the person voted against."""

# ╔═╡ 26c37043-a8dd-4557-898f-1cc7b9f14347
predicted_vote = ifelse.(predict(modlog4) .> 0.5, 1, 0);

# ╔═╡ 1d4054b4-95f7-4d2d-a681-41fe034f3a30
freqtable(predicted_vote)

# ╔═╡ 49af4ab8-566c-4aa0-a087-cf5ee75c4c17
proptable(predicted_vote)

# ╔═╡ 9984c317-0df6-453c-95fe-482f8afcfe13
md"""
Are we doing better than chance? How can we know that?"""

# ╔═╡ 7c7632fb-dc5c-4905-a5a1-0068c5887e24
md"""
## Logistic Regression

### Prediction

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)

A confusion matrix is simply a display of the ground truth (rows) x the model's prediction (columns).

![confm](https://github.com/umbertomig/POLI175julia/blob/c9b0555e3e97778495bee72746aee43ddf3226d7/img/confm_wikipedia.png?raw=true)"""

# ╔═╡ d2f6e649-5998-4943-aba5-c875e391a084
freqtable(chile.voteyes, predicted_vote)

# ╔═╡ 8a17f071-d23d-441d-846c-1a3f3ae80487
md"""
### Prediction

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)

|  | **Predicted: 0** | **Predicted: 1** |
|---|---|---|
| **Actual: 0** | True Negative (tn) | False Positive (fp) |
| **Actual: 1** | False Negative (fn) | True Positive (tp) |

[**Accuracy:**](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers) $$\dfrac{\text{correct predictions}}{\text{total observations}} \ = \ \dfrac{tp + tn}{tp + tn + fp + fn}$$

- High accuracy: lots of correct predictions!"""

# ╔═╡ 46105145-e274-4d4e-8eec-1e59ad5e8865
md"""
### Measuring Performance

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)

|  | **Predicted: 0** | **Predicted: 1** |
|---|---|---|
| **Actual: 0** | True Negative (tn) | False Positive (fp) |
| **Actual: 1** | False Negative (fn) | True Positive (tp) |

[**Precision**](https://en.wikipedia.org/wiki/Precision_and_recall): $$\dfrac{\text{true positives}}{\text{total predicted positive}} \ = \ \dfrac{tp}{tp + fp}$$

- High precision: low false-positive rates.
"""

# ╔═╡ 6a10fa30-4d1e-4805-a49c-9b376871c23c
md"""
### Measuring Performance

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)

|  | **Predicted: 0** | **Predicted: 1** |
|---|---|---|
| **Actual: 0** | True Negative (tn) | False Positive (fp) |
| **Actual: 1** | False Negative (fn) | True Positive (tp) |

[**Recall**](https://en.wikipedia.org/wiki/Precision_and_recall): $$\dfrac{\text{true positives}}{\text{total actual positive}} \ = \ \dfrac{tp}{tp + fn}$$

- High recall: low false-negative rates.
"""

# ╔═╡ 4e4414ca-de4f-4485-b9b2-cf95d1f80b21
md"""
### Measuring Performance

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)

|  | **Predicted: 0** | **Predicted: 1** |
|---|---|---|
| **Actual: 0** | True Negative (tn) | False Positive (fp) |
| **Actual: 1** | False Negative (fn) | True Positive (tp) |

[**Precision**](https://en.wikipedia.org/wiki/Precision_and_recall): $$\dfrac{\text{true positives}}{\text{total predicted positive}} \ = \ \dfrac{tp}{tp + fp}$$

[**Recall**](https://en.wikipedia.org/wiki/Precision_and_recall): $$\dfrac{\text{true positives}}{\text{total actual positive}} \ = \ \dfrac{tp}{tp + fn}$$

[**F1-Score**](https://en.wikipedia.org/wiki/F-score):

$$ \text{F1} \ = \ 2 \times \dfrac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}  \ = \ \dfrac{2 \ tp}{2 \ tp + fp + fn} $$"""

# ╔═╡ 9b8382d5-2115-4b95-b6c8-d8677043f123
md"""
## Logistic Regression

### Prediction

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)"""

# ╔═╡ 439fd0f5-1b5e-42c5-bf60-1805793ae3d1
conf_mat = ConfusionMatrix(chile.voteyes, predicted_vote)

# ╔═╡ 4e72a675-a445-427a-99f4-0c246ea76f9b
accuracy(conf_mat)

# ╔═╡ dabe9e3c-0bf7-4e6f-b121-ea44c231bb83
precision(conf_mat)

# ╔═╡ 78009c10-bc72-4eb7-833e-59c2260421ea
md"""
## Logistic Regression

### Prediction

[**Confusion Matrix**](https://en.wikipedia.org/wiki/Confusion_matrix)"""

# ╔═╡ a0190c3a-2f7e-4b45-9c0b-757e6a3e5fb8
recall(conf_mat)

# ╔═╡ cd1f24c5-ecf8-4c4c-b4cb-071ae27ce92f
f1_score(conf_mat)

# ╔═╡ 44b0d0ed-b127-4931-acfa-fad6bfd37382
md"""
## Logistic Regression

### Prediction

[**ROC Curve**](https://en.wikipedia.org/wiki/Receiver_operating_characteristic): Plots the true positive rates x false positive rates at each threshold.

The higher the area covered by the blue part (AUC, i.e., area under the curve), the better."""

# ╔═╡ e8c62e4a-aaff-4e15-a446-d156c6077898
rocplot(chile.voteyes, 
    predict(modlog4), 
    label = "vote ~ log(pop) + log(inc) + age", 
    diagonal = true)

# ╔═╡ f993fe74-d7e2-44e3-9ee7-939b15cd5446
md"""
## Why not run a Linear Regression?

You could ask this very valid question. And my answer here differs a bit from the book.

**Suggestion 01:**

If you want to **measure a treatment effect**, or any other fitting where **explanation trumps prediction**, go with the linear regression.
- Easy to explain to a lay audience.

- Good polynomial expansion around the ATE.

- But there is a need for a careful design (in Causal Inference, the design is more important than the statistical method!).

- Interaction terms are just partial derivatives of the fitted equation (no partial of the link function using chain rule)."""

# ╔═╡ c0d5c364-3d6c-4fe2-993d-099458b88a53
md"""
## Why not run a Linear Regression?

**Suggestion 02:**

If you want to **predict outcomes**, go with a classification model appropriate for your target variable unit.

- You are not going to do `weird` prediction.

- You have a marginal efficiency gain (in terms of Standard Errors).

- If you have an ordered target variable, your model does *look like more meaningful*.

- You should be careful when interpreting interaction terms."""

# ╔═╡ 798bd29a-8359-4903-ad67-80284198ef98
md"""
## Why not run a Linear Regression?

**Suggestion 03:**

Be **careful when you have discrete nominal variation in your target variable**:

- Binary outcome: Linear Regression and Linear Discriminant Analysis are the same.

- Three or more categories, like the `vote` in the Chilean dataset messes up badly with things."""

# ╔═╡ 8558310d-d56f-47ec-ac43-7b690fee88c7
md"""
# Questions?"""

# ╔═╡ f8926689-be63-494f-a1c4-be29b99137a2
md"""
# See you next class
"""

# ╔═╡ Cell order:
# ╟─f749ff71-2d26-4dec-8f6b-7f1353768587
# ╟─f99397c2-1b39-458a-9b02-bbd6ca7c6fd5
# ╠═62a6c9a3-4ceb-4706-8f69-40d79ca105cb
# ╟─787f3195-c449-4686-acea-925e90e40973
# ╠═3df02c92-1d6e-4a85-8f81-ad87eef43d79
# ╟─5e779878-468b-4f81-a532-8e76fb5da797
# ╟─6d2a95eb-0662-4854-b27d-6e8a82c09b39
# ╟─d0ac4d2c-10f8-46ac-a625-f6b03b9e8ffd
# ╟─29d8ecb7-4cce-4806-9dab-89f3c6f1dd37
# ╠═df57342e-7834-41d1-b691-820316c8d5e9
# ╟─82243c55-ea4e-48fb-a6d6-73e88a844532
# ╠═dec2b960-9e14-4f7a-8432-69b27ed43f44
# ╠═2c931a1e-bcbe-4ac1-8af1-cf69c7336b6a
# ╟─e2b698c5-6ac1-4000-8edf-a9544a35dfed
# ╟─35830a4e-8594-4352-922e-0c3bde3bd525
# ╟─e994c99a-491f-4c93-bbe4-87ef873b4e72
# ╟─7e889ac3-5293-448f-970f-860fd74edcb7
# ╟─164b7088-314f-4ad9-adf8-e7d8f1fb937a
# ╟─26102ac3-f9d6-412c-bc22-e1b7a99d35d5
# ╟─77f96bb5-3f2e-463a-928c-e036971971dc
# ╠═a3ddbf90-5314-458f-badb-fd49ac3bcc20
# ╟─dd37ea65-0307-48fe-bd8a-993ad4024d72
# ╟─cd081d49-a5fd-4e62-96a3-58c8bb2bde1d
# ╠═af8e439a-8acd-42d7-9751-753ef2b549e3
# ╟─45ea2d28-b9da-4b81-aa31-26b927c7b4f0
# ╟─73c1dc27-a4f4-4d07-8491-7504eb6f8d97
# ╠═068b8878-e84d-4b78-96b9-834d9a3d2140
# ╟─db16c15b-da56-4b2f-8d58-449fc93d5147
# ╠═a7456777-308c-4774-afa6-55f66b58f78e
# ╟─540ba93f-5073-469a-b409-5aa4acb6e665
# ╠═f89cc1c6-0afe-4179-a3b5-d29567b7901c
# ╟─99de1ab8-9fc5-4cbf-b600-d0ab99a953ea
# ╠═f9d4399b-b255-4319-ad83-3e24028498b3
# ╟─de595011-ae08-403e-aa0a-23341c6b6289
# ╠═282e665e-704a-455f-90a6-a3e947b4fbab
# ╟─6d3120a0-3a7b-4f29-978f-078713dc3340
# ╠═c77d51e2-3e5c-4c2e-a994-3855927cf83e
# ╟─77358e9b-d2e4-4490-af18-ac9f572ca876
# ╠═b67db2ce-5d11-4a62-90c4-b42d69618459
# ╠═6c0e975e-f696-41ae-b56d-ca43d26f7d6a
# ╟─97bb48e4-26c2-4b7e-904c-fad63d399ae0
# ╠═5697d8af-5f02-451c-873d-0cbb3241d15c
# ╟─3d455dee-090e-4014-9e09-12c3a7ce3672
# ╠═9af09918-9e7e-4865-8606-28a9e6629608
# ╟─9d8a601f-48fd-4447-90df-b8ccd7648f21
# ╟─b064e023-29c4-4832-be0d-988065de253f
# ╠═12ca9c1a-9f3c-4569-aafe-3dbf2b8e885c
# ╟─c17a10c6-994e-407f-bbaf-ddc18b23e39e
# ╠═646d785e-f8ff-439e-87b9-02a0e27f024e
# ╟─3d3a253f-5a3c-4f3b-9a3f-9ce900bf8fe1
# ╟─29ac4033-a530-4b52-a703-95bd40393038
# ╠═50a9393c-4e21-44b8-9a57-93b68774cb41
# ╟─6fce3d3a-b38c-43e8-887b-9ef190808a8c
# ╟─36057635-934c-4226-93d7-7232c7bb4b4a
# ╠═39add574-43cc-4e32-bb64-32f9fb6e92e9
# ╠═b3905735-463e-4c3a-8757-9788f04a21b5
# ╟─5eb8f5aa-f2f7-480e-bb33-437ff7263efa
# ╟─86f6d905-2e43-4171-b840-2b262620166d
# ╠═b8bba81a-a388-4429-b461-be5aa9c28023
# ╟─be85936d-6da3-4fc0-a2d6-857c91096015
# ╠═26c37043-a8dd-4557-898f-1cc7b9f14347
# ╠═1d4054b4-95f7-4d2d-a681-41fe034f3a30
# ╠═49af4ab8-566c-4aa0-a087-cf5ee75c4c17
# ╟─9984c317-0df6-453c-95fe-482f8afcfe13
# ╟─7c7632fb-dc5c-4905-a5a1-0068c5887e24
# ╠═d2f6e649-5998-4943-aba5-c875e391a084
# ╟─8a17f071-d23d-441d-846c-1a3f3ae80487
# ╟─46105145-e274-4d4e-8eec-1e59ad5e8865
# ╟─6a10fa30-4d1e-4805-a49c-9b376871c23c
# ╟─4e4414ca-de4f-4485-b9b2-cf95d1f80b21
# ╟─9b8382d5-2115-4b95-b6c8-d8677043f123
# ╠═439fd0f5-1b5e-42c5-bf60-1805793ae3d1
# ╠═4e72a675-a445-427a-99f4-0c246ea76f9b
# ╠═dabe9e3c-0bf7-4e6f-b121-ea44c231bb83
# ╟─78009c10-bc72-4eb7-833e-59c2260421ea
# ╠═a0190c3a-2f7e-4b45-9c0b-757e6a3e5fb8
# ╠═cd1f24c5-ecf8-4c4c-b4cb-071ae27ce92f
# ╟─44b0d0ed-b127-4931-acfa-fad6bfd37382
# ╠═e8c62e4a-aaff-4e15-a446-d156c6077898
# ╟─f993fe74-d7e2-44e3-9ee7-939b15cd5446
# ╟─c0d5c364-3d6c-4fe2-993d-099458b88a53
# ╟─798bd29a-8359-4903-ad67-80284198ef98
# ╟─8558310d-d56f-47ec-ac43-7b690fee88c7
# ╟─f8926689-be63-494f-a1c4-be29b99137a2
