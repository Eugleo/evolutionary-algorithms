### A Pluto.jl notebook ###
# v0.16.4

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
	
	using Gadfly: set_default_plot_size, cm
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
function roulette_wheel_selection(population, scores; population_size)
    StatsBase.sample(population, StatsBase.Weights(scores), population_size)
end

# ╔═╡ 7288f286-cd88-4722-97e0-bb1864230ebf
population

# ╔═╡ c1c054e6-f4de-4caf-924e-d9de45bc0c4e
roulette_wheel_selection(population, [3, 1, 0, 0]; population_size = 4)

# ╔═╡ 2ae0395b-a29f-4f8a-ad14-be9049d45f16
function make_tournament_selection(; tournament_size, p)
	function tournament_selection(population, scores; population_size)
		new_population = []
		tournament_size_n = ceil(Int, tournament_size * population_size)
		for _ in 1:population_size
			sample_indices = sample(1:length(population), tournament_size_n)
			indices = sortperm(scores[sample_indices], rev=true)
			for i in indices
				if rand() <= p || i == indices[end]
					append!(new_population, population[sample_indices[i], :])
					break
				end
			end
		end
		new_population
	end
end

# ╔═╡ 3d9228ad-211f-494d-97bd-fe620bc0b9a7
population

# ╔═╡ 7fc5a464-572d-4ab4-8b54-14d8571c29bf
begin
	t_sel = make_tournament_selection(tournament_size=0.5, p=0.1)
	t_sel(population, [0, 1, 2, 3]; population_size=2)
end

# ╔═╡ 73096df4-1e2f-4a65-8c72-847fc86147ed
function make_elitism_selection(; elites, select)
	function elitism_selection(population, scores; population_size)
		elites_n = ceil(Int, population_size * elites)
		best_indices = partialsortperm(scores, 1:elites_n, rev=true)
		best = population[best_indices, :]
		rest =
			select(population, scores, population_size = population_size - elites_n)
		[best; rest]
	end
end

# ╔═╡ f96d96a2-7934-4897-aae4-dca99685e343
begin
	e_sel = make_elitism_selection(elites=0.75, select=roulette_wheel_selection)
	e_sel(population, [1, 1.1, 1, 1]; population_size=4)
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

# ╔═╡ 6f019da5-54ec-4479-bf91-03494551157d
function choose_item(individual, bin_weights, item_weights, compute_weight)
	bin = sample(Weights(compute_weight(bin_weights)))
	items = findall(b -> b == bin, individual)
	
	item = nothing
	if length(items) > 0
		item = sample(items, Weights(compute_weight(item_weights[items])))
	end
	
	bin, item
end

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

# ╔═╡ a7ba4c45-1370-42de-820d-5241944dd06f
function make_move_mutation(; weights, classes)
	function mutate(individual)
		heaviest = argmax(bin_weights(weights, individual; classes))
		index = sample(findall(i -> i == heaviest, individual))
		individual[index] = sample([i for i in 1:classes if i != heaviest])
		individual
	end
end

# ╔═╡ 6599dd50-4f1b-4687-a709-89cd93030dbb
begin
	m = make_move_mutation(weights = [30, 5, 5, 5, 15], classes=10)
	m([1, 2, 2, 2, 3])
end

# ╔═╡ 9673d5ea-b99b-4f0d-a579-5bec6946cde2
choose_item(
	[1, 2, 2, 2, 3], 
	bin_weights([30, 5, 5, 5, 15], [1, 2, 2, 2, 3], classes = 10), 
	[30, 5, 5, 5, 15], 
	xs -> (xs .+ 1)
)

# ╔═╡ c2db4609-7391-458d-b8ba-fc5f62bdc110
function make_exchange_mutation(; weights, classes)
	function mutate(individual)
		bws = bin_weights(weights, individual; classes)
		
		heavy_bin, heavy_item = 
			choose_item(individual, bws, weights, xs -> xs .+ 1)
		light_bin, light_item = 
			choose_item(individual, bws, weights, xs -> 1 ./ (xs .+ 1))
		
		if !isnothing(heavy_item)
			individual[heavy_item] = light_bin
		end
		
		if !isnothing(light_item)
			individual[light_item] = heavy_bin
		end
		
		individual
	end
end

# ╔═╡ bc5986f6-5ace-4b9c-a3b0-2f13c06c1e51
begin
	me = make_exchange_mutation(weights = [30, 5, 5, 5, 15], classes=10)
	me([1, 2, 2, 2, 3])
end

# ╔═╡ 8037a721-2db7-4ed3-877d-7a68667eb9c7
"""The fitness function"""
function fitness_difference(individual, weights; classes)
    bw = bin_weights(weights, individual; classes)
    1 / (maximum(bw) - minimum(bw) + 1)
end

# ╔═╡ 1e451410-9380-4583-a505-d9eb6f58a908
"""The fitness function"""
function fitness_variance(individual, weights; classes)
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
function experiment(
    data;
    repeats,
    max_gen,
    pop_size,
    cx_prob,
    mut_prob,
    k,
	selection,
	fitness,
	mutate_individual
)
	weights = read_weights(data)
	cross = population -> crossover(population, one_pt_crossover, cx_prob)
	mutate = population -> mutation(population, mutate_individual, mut_prob)

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
			selection,
		)

		insertcols!(scores_log, :run => run)
		append!(scores_logs, scores_log)
	end

	scores_logs
end

# ╔═╡ aad744e6-b07f-4078-b5bf-0efc30984aaf
function describe_experiment(; 
		mut_prob, 
		cx_prob, 
		selection_type,
		fitness_type,
		mutation_type,
		_...)
	join(
		Utilities.encode_parameters(; 
			mut_prob, mutation_type, cx_prob, selection_type, fitness_type
		), 
		", "
	)
end

# ╔═╡ 7c538d10-9aa2-40e2-ae08-35ee005626df
begin
    EASY = "../data/partition-easy.txt"
    HARD = "../data/partition-hard.txt"
end

# ╔═╡ c944a9eb-7a91-40e3-a8b8-c17ed05d3318
data = read_weights(EASY)

# ╔═╡ 4cef95ad-391b-4260-8695-d5cbdbd97266
function run_experiment(; 
		data, 
		selection_type, 
		fitness_type, 
		mutation_type,
		k,
		configuration...
	)
	
	if selection_type == "elitism+roulette"
		selection = make_elitism_selection(
			elites=0.05, 
			select=roulette_wheel_selection
		)
	elseif selection_type == "elitism+tournament"
		tournament_selection = make_tournament_selection(
			tournament_size = 0.2,
			p = 0.75
		)
		selection = make_elitism_selection(
			elites = 0.05, 
			select = tournament_selection
		)
	else
		selection = roulette_wheel_selection
	end
	
	fitness = fitness_type == "variance" ? fitness_variance : fitness_difference
	
	weights = read_weights(data)
	if mutation_type == "original"
		mutate_individual = (ind) -> flip_mutate(ind, 0.004, upper = k)
	elseif mutation_type == "move"
		mutate_individual = make_move_mutation(classes = k; weights)
	elseif mutation_type == "exchange"
		mutate_individual = make_exchange_mutation(classes = k; weights)
		
	end
	
	dataset = data === "easy" ? EASY : HARD
	
	experiment(dataset; selection, fitness, mutate_individual, k, configuration...)
end

# ╔═╡ bb6e5003-e3a2-4de9-896f-10b549c3f56c
begin
	set_default_plot_size(17cm, 12cm)
	configuration = (
		data = ["easy"],
		repeats=[12],
		max_gen=[3000] ,
		pop_size=[100],
		cx_prob=[0.1, 0.2],
		mut_prob=[0.12], # 0.25
		# mut_flip_prob=[0.004],
		k=[10],
		selection_type=["elitism+tournament"],
		mutation_type=["original", "move", "exchange"],
		fitness_type=["variance"]
	)
	results, img = Utilities.plot_experiments(
		run_experiment,
		describe_experiment;
		path = "../out/3-set-partition",
		cache=false,
		metric = "objective",
		ranking = "lowest",
		configuration,
		log=true
	)
	img
end

# ╔═╡ 83b7dfd7-631a-4955-b13c-4055be39bea8
@chain results begin
	@subset(:metric .== "objective")
	@orderby(:score, :generation)
end

# ╔═╡ 8bbf871d-a5bb-43bd-8f0c-89ad8997e3dc
if true
	Utilities.save_plot(img, "../out/3-set-partition/comparison.png", 15cm, 10cm)
end

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
# ╠═7288f286-cd88-4722-97e0-bb1864230ebf
# ╠═c1c054e6-f4de-4caf-924e-d9de45bc0c4e
# ╠═2ae0395b-a29f-4f8a-ad14-be9049d45f16
# ╠═3d9228ad-211f-494d-97bd-fe620bc0b9a7
# ╠═7fc5a464-572d-4ab4-8b54-14d8571c29bf
# ╠═73096df4-1e2f-4a65-8c72-847fc86147ed
# ╠═f96d96a2-7934-4897-aae4-dca99685e343
# ╠═881bb2bb-2d46-41bd-a272-8da9c7be29ee
# ╠═d853c8e6-a0d2-4cf5-89cc-001a76d57a80
# ╠═546df62f-6e46-4252-a40e-25d7d68cc12c
# ╠═ccd98ecf-bb2f-4efe-bfe4-3836e9356cb5
# ╠═00181497-2248-477f-987f-7c6f000eb3fd
# ╠═580a048d-9848-4dac-8e60-6b7686b949be
# ╠═7493dd23-2eb3-466e-aafe-7e9131add024
# ╠═a7ba4c45-1370-42de-820d-5241944dd06f
# ╠═6599dd50-4f1b-4687-a709-89cd93030dbb
# ╠═6f019da5-54ec-4479-bf91-03494551157d
# ╠═9673d5ea-b99b-4f0d-a579-5bec6946cde2
# ╠═c2db4609-7391-458d-b8ba-fc5f62bdc110
# ╠═bc5986f6-5ace-4b9c-a3b0-2f13c06c1e51
# ╠═b573446d-b3b1-4655-8062-4d23c02e6073
# ╠═888210b4-3a45-41b6-b38f-83950f44e256
# ╠═709fd8a0-969b-49b3-8b67-6d8de51a7a17
# ╠═76ba3116-4863-4cc2-b53d-73dd2a3709f2
# ╟─b5e01ac0-8ea2-469d-8faf-be796710d3e7
# ╠═9122c3d6-a624-4705-8a72-132d715aceec
# ╠═8037a721-2db7-4ed3-877d-7a68667eb9c7
# ╠═1e451410-9380-4583-a505-d9eb6f58a908
# ╠═c3770f9c-ce0b-4879-ac94-5e8eab7292f3
# ╠═c163971c-a228-42e2-a9f9-c24337ed2138
# ╠═aad744e6-b07f-4078-b5bf-0efc30984aaf
# ╠═4cef95ad-391b-4260-8695-d5cbdbd97266
# ╠═7c538d10-9aa2-40e2-ae08-35ee005626df
# ╠═bb6e5003-e3a2-4de9-896f-10b549c3f56c
# ╠═83b7dfd7-631a-4955-b13c-4055be39bea8
# ╠═8bbf871d-a5bb-43bd-8f0c-89ad8997e3dc
