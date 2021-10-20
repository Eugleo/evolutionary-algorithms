module Utilities

export compute_statistics, with_serialization

using Statistics
using DataFrames
using DataFramesMeta
using Serialization
using Gadfly

# Dostane dataframe se sloupci
# configuration: konfigurace experimentu
# run: číslo běhu simulace v daném experimentu
# generation: číslo generace v rámci daného běhu
# individual: jedinec worst, best, average, hodnocen podle hodnoty fitness
# fitness: hodnota fitness daného jednice
# objective: hodnota objective daného jedince

function compute_statistics(results)
    @chain results begin
        groupby([:configuration, :evaluations, :individual, :metric])
        @combine(
            min = Statistics.minimum(:score),
            q1 = quantile(:score, 0.25),
            mean = Statistics.mean(:score),
            q3 = quantile(:score, 0.75),
            max = Statistics.maximum(:score)
        )
    end
end

function encode_parameters(; parameters...)
    ["$(String(name)[1])=$(value)" for (name, value) in pairs(parameters)]

end

function with_serialization(experiment; path_base, parameters...)
    filename = join(encode_parameters(parameters...), "_") * ".jls"
    path = joinpath(path_base, filename)

    if isfile(path)
        deserialize(path)
    else
        result = experiment(parameters...)
        serialize(path, result)
        result
    end
end

function run_experiments(experiment; description, path_base, cache = true, parameters...)
    results = DataFrame()
    for combination in Base.product(values(parameters)...)
        config = zip(keys(parameters), combination)
        if cache
            result = with_serialization(experiment; path_base, config...)
        else
            result = experiment(; config...)
        end
        insertcols!(result, :configuration => description(; config...))
        append!(results, result)
    end
    results
end

function plot_experiments(experiment; path_base, description, cache = true, parameters...)
    results = run_experiments(experiment; path_base, description, cache, parameters...)

    plot(
        @subset(
            Utilities.compute_statistics(results),
            :metric .== "objective",
            :individual .== "best"
        ),
        x = :evaluations,
        ymin = :q1,
        y = :mean,
        ymax = :q3,
        color = :configuration,
        Geom.line,
        Geom.ribbon,
        Scale.discrete_color(),
        Theme(key_position = :top),
    )
end


function with_serialization(experiment; path_base, parameters...)
    filename = []
    for (name, value) in pairs(parameters)
        push!(filename, "$(encode_string(name))=$(value)")
    end

    path = joinpath(path_base, join(filename, "_") * ".jls")

    if isfile(path)
        deserialize(path)
    else
        result = experiment(parameters...)
        serialize(path, result)
        result
    end
end

"""Implements the evolutionary algorithm
	arguments:
	- `pop_size`: the initial population
	- `max_gen`: maximum number of generation
	- `fitness`: fitness function (takes individual as argument and returns FitObjPair)
	- `operators`: list of genetic operators (functions of type Population -> Population)
	- `select`: mating selection (function with three arguments - population, fitness values, number of individuals to select; returning the selected population)"""
function evolve(population, pop_size, generation_count, metrics, operators, select)
    metric_names = keys(metrics)
    evaluations = 0
    scores_log = []

    for generation = 1:generation_count
        metric_values = [[metric(ind) for ind in population] for metric in metrics]
        scores = (; zip(metric_names, metric_values)...)
        evaluations += length(population)

        for individual in ["worst", "average", "best"]
            append!(scores_log, summarize(generation, evaluations, scores, individual))
        end

        mating_pool = select(population, scores[1], count = pop_size)
        population = mate(mating_pool, operators)
    end

    population, DataFrame(scores_log)
end

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
        ) for (metric, values) in pairs(scores)
    ]
end

"""Applies a list of genetic operators (functions of type Population -> Population) to the population"""
function mate(population, operators)
    for op in operators
        population = op(population)
    end
    population
end

end
