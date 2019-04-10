using Random
using Distributions

N=1_000_000; ## population size
n=1000; ## random sample
logit_los(x) = exp(x) / (1+exp(x));
## population data
Random.seed!(123)
X₁ = rand(Normal(1,1),N);
X₂ = rand(Exponential(1), N);
α₁ = randn(N);
ϵ₁ = randn(N);

Y₂ = [rand(Bernoulli(p)) for p in logit_los.(X₁+X₂ + α₁ .+ 1)];
@time δ =[rand(Bernoulli(p)) for p in logit_los.(X₂)];
