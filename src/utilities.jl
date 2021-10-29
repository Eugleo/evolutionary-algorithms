module Utilities

export compute_statistics, with_serialization, save_plot, save_best

using Statistics
using DataFrames
using DataFramesMeta
using Serialization
using Gadfly

import Cairo
import Fontconfig
import CSV

# Dostane dataframe se sloupci
# configuration: konfigurace experimentu
# run: číslo běhu simulace v daném experimentu
# generation: číslo generace v rámci daného běhu
# individual: jedinec lowest, highest, average, hodnocen podle hodnoty fitness
# fitness: hodnota fitness daného jednice
# objective: hodnota objective daného jedince

function compute_statistics(results)
    @chain results begin
        groupby([:configuration, :evaluations, :ranking, :metric])
        @combine(
            min = Statistics.minimum(:score),
            q1 = quantile(:score, 0.25),
            mean = Statistics.mean(:score),
            q3 = quantile(:score, 0.75),
            max = Statistics.maximum(:score),
        )
    end
end

function encode_parameters(; parameters...)
    [
        "$(join([s[1] for s in split(String(name), "_")]))=$(value)" for
        (name, value) in pairs(parameters)
    ]

end

function with_serialization(experiment; path, configuration)
    filename = join(encode_parameters(; configuration...), "_") * ".jls"
    path = joinpath(path, filename)

    if isfile(path)
        deserialize(path)
    else
        result = experiment(; configuration...)
        serialize(path, result)
        result
    end
end

function run_experiments(experiment, descriptor; configuration, path, cache = true)
    results = DataFrame()
    for combination in Base.product(values(configuration)...)
        run_config = zip(keys(configuration), combination)
        if cache
            result = with_serialization(experiment; path, configuration = run_config)
        else
            result = experiment(; run_config...)
        end
        insertcols!(
            result,
            :configuration => descriptor(; run_config...),
            :configuration_raw => Ref(collect(run_config)),
        )
        append!(results, result)
    end
    results
end

function plot_experiments(
    experiment,
    descriptor;
    path,
    configuration,
    metric,
    ranking,
    cache = true,
    log = false,
)
    data = run_experiments(experiment, descriptor; path, cache, configuration)
    img = plot_data(data; metric, ranking, log)
    data, img
end

function plot_data(data; metric, ranking, log = false)
    plot(
        @subset(
            Utilities.compute_statistics(data),
            :metric .== metric,
            :ranking .== ranking
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
        log ? Scale.y_log10 : Scale.y_continuous,
    )
end

function save_plot(img, path, width = 30cm, height = 15cm)
    draw(PNG(path, width, height), img)
end

function save_best(data, path)
    @chain data begin
        @subset(:metric .== "objective", :ranking .== "lowest", :generation .% 1000 .== 0)
        @orderby(:generation)
        @select(:generation, :score, :configuration, :individual)
        CSV.write(path, _)
    end
end

"""Implements the evolutionary algorithm
	arguments:
	- `pop_size`: the initial population
	- `max_gen`: maximum number of generation
	- `fitness`: fitness function (takes individual as argument and returns FitObjPair)
	- `operators`: list of genetic operators (functions of type Population -> Population)
	- `select`: mating selection (function with three arguments - population, fitness values,
    number of individuals to select; returning the selected population)"""
function evolve(population, pop_size, generation_count, metrics, operators, select)
    metric_names = keys(metrics)
    evaluations = 0
    scores_log = []

    for generation = 1:generation_count
        metric_values = [[metric(ind) for ind in population] for metric in metrics]
        scores = (; zip(metric_names, metric_values)...)
        evaluations += length(population)

        for ranking in ["lowest", "average", "highest"]
            summary = summarize(generation, population, evaluations, scores, ranking)
            append!(scores_log, summary)
        end

        mating_pool = select(population, scores[1], population_size = pop_size)
        population = mate(mating_pool, operators)
    end

    population, DataFrame(scores_log)
end

function summarize(generation, population, evaluations, scores, ranking)
    results = []
    for (metric, values) in pairs(scores)
        if ranking == "lowest"
            index = argmin(values)
            score = values[index]
            individual = population[index]
        elseif ranking == "average"
            score = mean(values)
            individual = nothing
        elseif ranking == "highest"
            index = argmax(values)
            score = values[index]
            individual = population[index]
        end

        obs = (
            metric = String(metric),
            generation = generation,
            evaluations = evaluations,
            ranking = ranking,
            score = score,
            individual = individual,
        )
        push!(results, obs)
    end
    results
end

"""Applies a list of genetic operators (functions of type Population -> Population) to the population"""
function mate(population, operators)
    for op in operators
        population = op(population)
    end
    population
end

end
