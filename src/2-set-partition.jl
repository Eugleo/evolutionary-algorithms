### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 18b30706-2f5a-11ec-0ef2-a7dcb21aa9ca
begin
    import Pkg
    Pkg.activate(Base.current_project())
end

# ╔═╡ 1ff40249-0294-43e5-912e-3d94987ff885
begin
	using Serialization
    using DataFrames
    using DataFramesMeta
    using StatsBase
    using Statistics
	
    include("./utilities.jl")
    using .Utilities
end

# ╔═╡ d5c3ead4-7739-40fe-978b-1d2796e29223
md"""
# Set partition problem

First, we need to setup the environment.
"""

# ╔═╡ 7376081d-a73e-4bff-bc0b-b609c3bd2c32
md"""
## Loading the data
"""

# ╔═╡ 1c86ce8c-7aaa-4f72-a679-d87a4756d021
"""Reads the input set of values of objects"""
function read_weights(filename)
    [parse(Int, line) for line in readlines("../data/partition-easy.txt")]
end

# ╔═╡ 130d440c-71dd-4e4e-8cd0-e390ff99f8af
"""Creates the individual"""
function create_individual(individual_length; upper)
    [rand(1:upper) for _ in 1:individual_length]
end

# ╔═╡ 6ed846ba-c5ff-459c-8ece-2a7b557dfcc9
individual = create_individual(10, upper=10)

# ╔═╡ 4a3540b8-e881-48bb-bf1f-4b90cb627cc3
"""Creates the population using the create individual function"""
function create_population(population_size, create)
    [create() for _ in 1:population_size]
end

# ╔═╡ dcc196c6-ab9c-4b3f-a611-a438744c8697
population = create_population(4, () -> create_individual(5, upper=10))

# ╔═╡ 8e8ec7aa-4685-4b1f-b093-088f11301d61
md"""
## Genetic operators: selection, mutation, crossover
"""

# ╔═╡ cbe39a0a-45d2-46b8-b5ad-76e1d7252947
"""Roulette wheel selection"""
function roulette_wheel_selection(population, scores; count)
    StatsBase.sample(population, StatsBase.Weights(scores), count)
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
function flip_mutate(individual, prob; upper)
    [rand() < prob ? rand(1:upper) : i for i in individual]
end

# ╔═╡ 580a048d-9848-4dac-8e60-6b7686b949be
population[1]

# ╔═╡ 7493dd23-2eb3-466e-aafe-7e9131add024
flip_mutate(population[1], 0.5, upper=10)

# ╔═╡ b573446d-b3b1-4655-8062-4d23c02e6073
"""Applies the mutate function (implementing the mutation of a single individual) to the whole population with probability mut_prob)"""
function mutation(population, mutate, prob)
    [rand() < prob ? mutate(i) : i for i in population]
end

# ╔═╡ 888210b4-3a45-41b6-b38f-83950f44e256
population

# ╔═╡ 709fd8a0-969b-49b3-8b67-6d8de51a7a17
mutation(population, (i) -> flip_mutate(i, 0.1, upper=10), 1)

# ╔═╡ 76ba3116-4863-4cc2-b53d-73dd2a3709f2
population

# ╔═╡ b5e01ac0-8ea2-469d-8faf-be796710d3e7
md"
## The algorithm itself
"

# ╔═╡ 9122c3d6-a624-4705-8a72-132d715aceec
"""Computes the bin weights.
Bins are the indices of bins into which the object belongs."""
function bin_weights(weights, bins; classes)
    bw = zeros(classes)
    for (w, b) in zip(weights, bins)
        bw[b] += w
    end
    bw
end

# ╔═╡ 8037a721-2db7-4ed3-877d-7a68667eb9c7
"""The fitness function"""
function fitness(individual, weights; classes)
    bw = bin_weights(weights, individual; classes)
    1 / (Statistics.var(bw) + 1)
end

# ╔═╡ c3770f9c-ce0b-4879-ac94-5e8eab7292f3
"""The objective function"""
function objective(individual, weights; classes)
    bw = bin_weights(weights, individual; classes)
    maximum(bw) - minimum(bw)
end

# ╔═╡ c163971c-a228-42e2-a9f9-c24337ed2138
function run_experiment(
    data;
    repeats,
    max_gen,
    pop_size,
    cx_prob,
    mut_prob,
    mut_flip_prob,
    k
)
	weights = read_weights(data)

	cross = population -> 
		crossover(population, one_pt_crossover, cx_prob)
	mutate = population ->
		mutation(
			population, 
			(ind) -> flip_mutate(ind, mut_flip_prob, upper=k), mut_prob
		)

	scores_logs = DataFrame()
	for run = 1:repeats
		population = create_population(
			pop_size, 
			() -> create_individual(length(weights), upper=k)
		)

		population, scores_log = Utilities.evolve(
			population,
			pop_size,
			max_gen,
			(
				fitness = (ind) -> fitness(ind, weights; classes=k),
				objective = (ind) -> objective(ind, weights; classes=k),
			),
			[cross, mutate],
			roulette_wheel_selection,
		)

		insertcols!(scores_log, :run => run)
		append!(scores_logs, scores_log)
	end

	scores_logs
end

# ╔═╡ 7c538d10-9aa2-40e2-ae08-35ee005626df
begin
    K = [10] #number of piles
    POP_SIZE = [100] # population size
    MAX_GEN = [500] # maximum number of generations
    MUT_FLIP_PROB = [0.1] # probability of chaninging value during mutation
	REPEATS = [4]
	CX_PROB = [1]
	MUT_PROB = [0.05]
    EASY = "../data/partition-easy.txt"
    HARD = "../data/partition-hard.txt"
end

# ╔═╡ c944a9eb-7a91-40e3-a8b8-c17ed05d3318
data = read_weights(EASY)

# ╔═╡ bb6e5003-e3a2-4de9-896f-10b549c3f56c
Utilities.plot_experiments(
	(; data, kwargs...) -> run_experiment(data === "easy" ? EASY : HARD; kwargs...),
	path_base = "../out/2-set-partition",
	description = (; mut_prob, mut_flip_prob, cx_prob, _...) -> 
		join(Utilities.encode_parameters(; mut_prob, mut_flip_prob, cx_prob), ", "),
	cache=false,
	data = ["easy"],
	repeats=REPEATS,
	max_gen=MAX_GEN,
	pop_size=POP_SIZE,
	cx_prob=CX_PROB,
	mut_prob=MUT_PROB,
	mut_flip_prob=MUT_FLIP_PROB,
	k=K
)

# ╔═╡ Cell order:
# ╟─d5c3ead4-7739-40fe-978b-1d2796e29223
# ╠═18b30706-2f5a-11ec-0ef2-a7dcb21aa9ca
# ╠═1ff40249-0294-43e5-912e-3d94987ff885
# ╟─7376081d-a73e-4bff-bc0b-b609c3bd2c32
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
# ╠═76ba3116-4863-4cc2-b53d-73dd2a3709f2
# ╟─b5e01ac0-8ea2-469d-8faf-be796710d3e7
# ╠═9122c3d6-a624-4705-8a72-132d715aceec
# ╠═8037a721-2db7-4ed3-877d-7a68667eb9c7
# ╠═c3770f9c-ce0b-4879-ac94-5e8eab7292f3
# ╠═c163971c-a228-42e2-a9f9-c24337ed2138
# ╠═7c538d10-9aa2-40e2-ae08-35ee005626df
# ╠═bb6e5003-e3a2-4de9-896f-10b549c3f56c
