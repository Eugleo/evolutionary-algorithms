### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 18b30706-2f5a-11ec-0ef2-a7dcb21aa9ca
begin
    import Pkg
    Pkg.activate(Base.current_project())
end

# ╔═╡ 1ff40249-0294-43e5-912e-3d94987ff885
begin
    using PlutoUI
    using Memoize
    using DataFrames
    using DataFramesMeta
    using Gadfly
    using StatsBase
    include("./utilities.jl")
    using .Utilities
end

# ╔═╡ d5c3ead4-7739-40fe-978b-1d2796e29223
md"""
# Set partition problem

First, we need to setup the environment.
"""

# ╔═╡ 7c538d10-9aa2-40e2-ae08-35ee005626df
begin
    K = 10 #number of piles
    POP_SIZE = 100 # population size
    MAX_GEN = 500 # maximum number of generations
    MUT_FLIP_PROB = 0.1 # probability of chaninging value during mutation
    OUT_DIR = "partition" # output directory for logs
    EXP_ID = "default" # the ID of this experiment (used to create log names)

    EASY = "../data/partition-easy.txt"
    HARD = "../data/partition-hard.txt"
end

# ╔═╡ 7376081d-a73e-4bff-bc0b-b609c3bd2c32
md"""
## Loading the data
"""

# ╔═╡ 1c86ce8c-7aaa-4f72-a679-d87a4756d021
"""Reads the input set of values of objects"""
function read_weights(filename)
    [parse(Int, line) for line in readlines("../data/partition-easy.txt")]
end

# ╔═╡ c944a9eb-7a91-40e3-a8b8-c17ed05d3318
data = read_weights(EASY)

# ╔═╡ 130d440c-71dd-4e4e-8cd0-e390ff99f8af
"""Creates the individual"""
function create_individual(individual_length)
    [rand(1:K) for _ = 1:individual_length]
end

# ╔═╡ 6ed846ba-c5ff-459c-8ece-2a7b557dfcc9
individual = create_individual(10)

# ╔═╡ 4a3540b8-e881-48bb-bf1f-4b90cb627cc3
"""Creates the population using the create individual function"""
function create_population(population_size, create)
    [create() for _ = 1:population_size]
end

# ╔═╡ dcc196c6-ab9c-4b3f-a611-a438744c8697
population = create_population(4, () -> create_individual(5))

# ╔═╡ 8e8ec7aa-4685-4b1f-b093-088f11301d61
md"""
## Genetic operators: selection, mutation, crossover
"""

# ╔═╡ cbe39a0a-45d2-46b8-b5ad-76e1d7252947
"""Roulette wheel selection"""
function roulette_wheel_selection(population, scores, k)
    StatsBase.sample(population, StatsBase.Weights(scores), k)
end

# ╔═╡ 881bb2bb-2d46-41bd-a272-8da9c7be29ee
"""Implements one-point crossover of two individuals"""
function one_pt_crossover(p1, p2)
    point = rand(1:length(p1))
    o1 = [p1[1:point]; p2[point+1:end]]
    o2 = [p2[1:point]; p1[point+1:end]]
    o1, o2
end

# ╔═╡ d853c8e6-a0d2-4cf5-89cc-001a76d57a80
[population[1], population[2]]

# ╔═╡ 546df62f-6e46-4252-a40e-25d7d68cc12c
one_pt_crossover(population[1], population[2])

# ╔═╡ ccd98ecf-bb2f-4efe-bfe4-3836e9356cb5
"""Applies the cross function (implementing the crossover of two individuals) to the whole population (with probability prob)"""
function crossover(population, cross, prob)
    offsprings = []
    for (p1, p2) in zip(population[1:2:end], population[2:2:end])
        if rand() < prob
            o1, o2 = cross(p1, p2)
        else
            o1, o2 = p1, p2
        end
        push!(offsprings, o1)
        push!(offsprings, o2)
    end
    offsprings
end

# ╔═╡ 00181497-2248-477f-987f-7c6f000eb3fd
"""Implements the "bit-flip" mutation of one individual"""
function flip_mutate(individual, prob, upper)
    [rand() < prob ? rand(1:upper) : i for i in individual]
end

# ╔═╡ 580a048d-9848-4dac-8e60-6b7686b949be
population[1]

# ╔═╡ 7493dd23-2eb3-466e-aafe-7e9131add024
flip_mutate(population[1], 0.5, K)

# ╔═╡ b573446d-b3b1-4655-8062-4d23c02e6073
"""Applies the mutate function (implementing the mutation of a single individual) to the whole population with probability mut_prob)"""
function mutation(population, mutate, prob)
    [rand() < prob ? mutate(i) : i for i in population]
end

# ╔═╡ 888210b4-3a45-41b6-b38f-83950f44e256
population

# ╔═╡ 709fd8a0-969b-49b3-8b67-6d8de51a7a17
mutation(population, (i) -> flip_mutate(i, 0.1, K), 1)

# ╔═╡ a2d7e84c-abd7-488d-b8b2-bdcbceb20347
"""Applies a list of genetic operators (functions of type Population -> Population) to the population"""
function mate(population, operators)
    for op in operators
        population = op(population)
    end
    population
end

# ╔═╡ 76ba3116-4863-4cc2-b53d-73dd2a3709f2
population

# ╔═╡ aaeee079-c5dc-4117-a781-285d5be4b0a8
mate(
    population,
    [
        p -> crossover(p, one_pt_crossover, 1),
        p -> mutation(population, (i) -> flip_mutate(i, 0.1, K), 1),
    ],
)

# ╔═╡ b5e01ac0-8ea2-469d-8faf-be796710d3e7
md"
## The algorithm itself
"

# ╔═╡ 9122c3d6-a624-4705-8a72-132d715aceec
"""Computes the bin weights.
Bins are the indices of bins into which the object belongs."""
function bin_weights(weights, bins)
    bw = zeros(K)
    for (w, b) in zip(weights, bins)
        bw[b] += w
    end
    bw
end

# ╔═╡ 8037a721-2db7-4ed3-877d-7a68667eb9c7
"""The fitness function"""
function fitness(individual, weights)
    bw = bin_weights(weights, individual)
    fitness = 1 / (maximum(bw) - minimum(bw) + 1)
end

# ╔═╡ c3770f9c-ce0b-4879-ac94-5e8eab7292f3
"""The objective function"""
function objective(individual, weights)
    bw = bin_weights(weights, individual)
    objective = maximum(bw) - minimum(bw)
end

# ╔═╡ e55e797d-24f5-44d8-8ff0-f3049c8274aa
function summarize(generation, evaluations, scores, individual)
    if individual == "worst"
        agg = minimum
    elseif individual == "average"
        agg = mean
    elseif individual == "best"
        agg = maximum
    end

    [
        (
            generation = generation,
            evaluations = evaluations,
            individual = individual,
            metric = String(metric),
            score = agg(values),
        )


        for (metric, values) in pairs(scores)
    ]


end

# ╔═╡ 35016aed-c5fa-4c32-ba27-1ccd20f305a9

"""Implements the evolutionary algorithm
	arguments:
	- `pop_size`: the initial population
	- `max_gen`: maximum number of generation
	- `fitness`: fitness function (takes individual as argument and returns FitObjPair)
	- `operators`: list of genetic operators (functions of type Population -> Population)
	- `mate_sel`: mating selection (function with three arguments - population, fitness values, number of individuals to select; returning the selected population)
	- `map_fn`: function to use to map fitness evaluation over the whole population (default is `map`)
	- `log`: a utils.Log structure to log the evolution run"""
function evolve(population, generation_count, metrics, operators, select)
    metric_names = keys(metrics)
    evaluations = 0
    scores_log = []

    for generation = 1:generation_count
        scores = (;
            zip(
                metric_names,
                [[metric(ind) for ind in population] for metric in metrics],
            )...,
        )

        evaluations += length(population)

        for individual in ["worst", "average", "best"]
            append!(scores_log, summarize(generation, evaluations, scores, individual))
        end


        mating_pool = select(population, scores[1], POP_SIZE)
        population = mate(mating_pool, operators)
    end

    population, DataFrame(scores_log)
end

# ╔═╡ c163971c-a228-42e2-a9f9-c24337ed2138
@memoize function run_experiment(
    data,
    repeats,
    max_gen,
    pop_size,
    cx_prob,
    mut_prob,
    mut_flip_prob,
    k,
)
    weights = read_weights(data)

    cross = population -> crossover(population, one_pt_crossover, cx_prob)
    mutate =
        population ->
            mutation(population, (ind) -> flip_mutate(ind, mut_flip_prob, k), mut_prob)

    scores_logs = DataFrame()
    Threads.@threads for run = 1:repeats
        population = create_population(pop_size, () -> create_individual(length(weights)))

        population, scores_log = evolve(
            population,
            max_gen,
            (
                fitness = (ind) -> fitness(ind, weights),
                objective = (ind) -> fitness(ind, weights),
            ),
            [cross, mutate],
            roulette_wheel_selection,
        )

        insertcols!(scores_log, :run => run)
        append!(scores_logs, scores_log)
    end

    scores_logs
end

# ╔═╡ 79de2ca3-0ada-4538-9de1-b44193f61adf
plot(
    @subset(
        Utilities.compute_statistics(scores_logs),
        :metric .== "objective",
        :individual .== "best"
    ),
    x = :evaluations,
    ymin = :q1,
    y = :mean,
    ymax = :q3,
    Geom.line,
    Geom.ribbon,
)

# ╔═╡ 6d427337-39ab-4317-8444-bcf5a0ee2a90
@bind REPEATS Slider(1:10)

# ╔═╡ 526b3307-476b-4d47-bf6c-fe42743b032d
REPEATS

# ╔═╡ 99345b9d-c778-41ca-8137-1a5f1bf4b1dc
@bind CX_PROB Slider(0:0.1:1)

# ╔═╡ 61ce3011-7093-486c-82f8-f581ddfcb4de
@bind MUT_PROB Slider(0:0.1:1)

# ╔═╡ 651f0c38-ecae-4cee-91bb-0596af3105c1
scores_logs =
    run_experiment(EASY, REPEATS, MAX_GEN, POP_SIZE, CX_PROB, MUT_PROB, MUT_FLIP_PROB, K)

# ╔═╡ 01f64ee2-2a79-417a-bc7a-439679e825e1
Utilities.compute_statistics(scores_logs)

# ╔═╡ cbe7a7e5-4f79-44cb-8e73-9504a616ddbf
scores_logs

# ╔═╡ Cell order:
# ╟─d5c3ead4-7739-40fe-978b-1d2796e29223
# ╠═18b30706-2f5a-11ec-0ef2-a7dcb21aa9ca
# ╠═1ff40249-0294-43e5-912e-3d94987ff885
# ╠═7c538d10-9aa2-40e2-ae08-35ee005626df
# ╠═7376081d-a73e-4bff-bc0b-b609c3bd2c32
# ╠═1c86ce8c-7aaa-4f72-a679-d87a4756d021
# ╠═c944a9eb-7a91-40e3-a8b8-c17ed05d3318
# ╠═130d440c-71dd-4e4e-8cd0-e390ff99f8af
# ╠═6ed846ba-c5ff-459c-8ece-2a7b557dfcc9
# ╠═4a3540b8-e881-48bb-bf1f-4b90cb627cc3
# ╠═dcc196c6-ab9c-4b3f-a611-a438744c8697
# ╟─8e8ec7aa-4685-4b1f-b093-088f11301d61
# ╠═cbe39a0a-45d2-46b8-b5ad-76e1d7252947
# ╠═881bb2bb-2d46-41bd-a272-8da9c7be29ee
# ╠═d853c8e6-a0d2-4cf5-89cc-001a76d57a80
# ╠═546df62f-6e46-4252-a40e-25d7d68cc12c
# ╠═ccd98ecf-bb2f-4efe-bfe4-3836e9356cb5
# ╠═00181497-2248-477f-987f-7c6f000eb3fd
# ╠═580a048d-9848-4dac-8e60-6b7686b949be
# ╠═7493dd23-2eb3-466e-aafe-7e9131add024
# ╠═b573446d-b3b1-4655-8062-4d23c02e6073
# ╠═888210b4-3a45-41b6-b38f-83950f44e256
# ╠═709fd8a0-969b-49b3-8b67-6d8de51a7a17
# ╠═a2d7e84c-abd7-488d-b8b2-bdcbceb20347
# ╠═76ba3116-4863-4cc2-b53d-73dd2a3709f2
# ╠═aaeee079-c5dc-4117-a781-285d5be4b0a8
# ╟─b5e01ac0-8ea2-469d-8faf-be796710d3e7
# ╠═9122c3d6-a624-4705-8a72-132d715aceec
# ╠═8037a721-2db7-4ed3-877d-7a68667eb9c7
# ╠═c3770f9c-ce0b-4879-ac94-5e8eab7292f3
# ╠═e55e797d-24f5-44d8-8ff0-f3049c8274aa
# ╠═35016aed-c5fa-4c32-ba27-1ccd20f305a9
# ╠═c163971c-a228-42e2-a9f9-c24337ed2138
# ╠═651f0c38-ecae-4cee-91bb-0596af3105c1
# ╠═79de2ca3-0ada-4538-9de1-b44193f61adf
# ╠═6d427337-39ab-4317-8444-bcf5a0ee2a90
# ╠═526b3307-476b-4d47-bf6c-fe42743b032d
# ╠═99345b9d-c778-41ca-8137-1a5f1bf4b1dc
# ╠═61ce3011-7093-486c-82f8-f581ddfcb4de
# ╠═01f64ee2-2a79-417a-bc7a-439679e825e1
# ╠═cbe7a7e5-4f79-44cb-8e73-9504a616ddbf
