using StatsBase
import Statistics

using DataFrames
using DataFramesMeta
using Gadfly

function create_individual(size)
    [rand((0, 1)) for _ = 1:size]
end

function create_random_population(individual_size, population_size)
    [create_individual(individual_size) for _ = 1:population_size]
end

function select(individuals, fitness_scores, population_size)
    sample(individuals, Weights(fitness_scores), population_size)
end

function crossover(pool, crossover_rate)
    offsprings = []
    for parents in zip(pool[1:2:end], pool[2:2:end])
        if rand() < crossover_rate
            append!(offsprings, cross(parents...))
        else
            append!(offsprings, parents)
        end
    end
    offsprings
end

function cross(a_parent, b_parent)
    point = rand(1:length(a_parent))
    offspring_a = [a_parent[1:point-1]; b_parent[point:end]]
    offspring_b = [b_parent[1:point-1]; a_parent[point:end]]
    offspring_a, offspring_b
end

function mutation(pool, mutation_rate)
    [mutate(parent, mutation_rate) for parent in pool]
end

function mutate(parent, mutation_rate)
    function point_mutation(x)
        if rand() < mutation_rate
            1 - x
        else
            x
        end
    end

    [point_mutation(x) for x in parent]
end

function mate(pool, mutation_rate, crossover_rate)
    mutation(crossover(pool, crossover_rate), mutation_rate)
end

function evolve(
    individual_size,
    population_size,
    generations,
    fitness,
    mutation_rate,
    crossover_rate,
)
    results = []
    population = create_random_population(individual_size, population_size)
    for generation = 1:generations
        fitness_scores = [fitness(individual) for individual in population]
        labeled = [(generation = generation, score = score) for score in fitness_scores]
        append!(results, labeled)

        mating_pool = select(population, fitness_scores, population_size)
        population = mate(mating_pool, mutation_rate, crossover_rate)
    end

    population, DataFrame(results)
end

function onemax_fitness(individual)
    sum(individual)
end

function flip_fitness_positions(individual)
    abs(sum(individual[1:2:end-1]) - sum(individual[2:2:end])) / (length(individual) / 2)
end

function flip_fitness_changes(individual)
    result = 0
    for i = 2:length(individual)
        if individual[i] != individual[i-1]
            result += 1
        end
    end
    result / (length(individual) - 1)
end


function run_experiment(;
    individual_size,
    population_size,
    generations,
    mutation_rates,
    crossover_rates,
    fitness_functions,
    n = 100,
)
    results = DataFrame()

    for iteration = 1:n
        for f in fitness_functions
            for mr in mutation_rates
                for cr in crossover_rates
                    _, scores =
                        evolve(individual_size, population_size, generations, f.fun, mr, cr)
                    insertcols!(
                        scores,
                        :iteration => iteration,
                        :configuration => (m = mr, c = cr, f = f.label),
                    )
                    append!(results, scores)
                end
            end
        end
    end

    @chain results begin
        groupby([:configuration, :iteration, :generation])
        @combine(
            :mean = Statistics.mean(:score),
            :min = Statistics.minimum(:score),
            :max = Statistics.maximum(:score),
            q1 = quantile(:score, 0.25),
            q3 = quantile(:score, 0.75)
        )
        groupby([:configuration, :generation])
        @combine(
            mean = Statistics.mean(:mean),
            :min = Statistics.mean(:min),
            :max = Statistics.mean(:max),
            q1 = Statistics.mean(:q1),
            q3 = Statistics.mean(:q3)
        )
        @transform(:fitness_evaluations = population_size * :generation)
        plot(
            x = :fitness_evaluations,
            y = :mean,
            ymin = :q1,
            ymax = :q3,
            color = :configuration,
            Geom.line,
            Geom.ribbon,
            Scale.color_discrete,
            Guide.title("Mean, Q1 and Q3 scores averaged over $n runs"),
            Guide.xlabel("Number of fitness function evaluations"),
            Guide.ylabel("Fitness score"),
            Guide.colorkey("Experiment configuration"),
        )
    end
end

# OneMax
run_experiment(
    individual_size = 25,
    population_size = 50,
    generations = 100,
    mutation_rates = [0.001, 0.01, 1],
    crossover_rates = [1, 0.1, 0.01],
    fitness_functions = [(label = "sum", fun = onemax_fitness)],
    n = 10,
)

# ALternating
run_experiment(
    individual_size = 25,
    population_size = 50,
    generations = 100,
    mutation_rates = [0.001],
    crossover_rates = [1],
    fitness_functions = [
        (label = "changes", fun = flip_fitness_changes),
        (label = "positions", fun = flip_fitness_positions),
    ],
    n = 100,
)
