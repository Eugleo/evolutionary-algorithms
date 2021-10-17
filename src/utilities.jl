module Utilities

using Statistics
using DataFrames
using DataFramesMeta

# Dostane dataframe se sloupci
# configuration: konfigurace experimentu
# run: číslo běhu simulace v daném experimentu
# generation: číslo generace v rámci daného běhu
# individual: jedinec worst, best, average, hodnocen podle hodnoty fitness
# fitness: hodnota fitness daného jednice
# objective: hodnota objective daného jedince
function compute_statistics(results)
    @chain results begin
        groupby([:evaluations, :individual, :metric])
        @combine(
            min = Statistics.minimum(:score),
            q1 = quantile(:score, 0.25),
            mean = Statistics.mean(:score),
            q3 = quantile(:score, 0.75),
            max = Statistics.maximum(:score)
        )
    end
end

# function compute_statistics(results)
#     @chain results begin
#         groupby([:configuration, :run, :generation])
#         @combine(
#             :mean = Statistics.mean(:score),
#             :min = Statistics.minimum(:score),
#             :max = Statistics.maximum(:score),
#             q1 = quantile(:score, 0.25),
#             q3 = quantile(:score, 0.75)
#         )
#         groupby([:configuration, :generation])
#         @combine(
#             mean = Statistics.mean(:mean),
#             :min = Statistics.mean(:min),
#             :max = Statistics.mean(:max),
#             q1 = Statistics.mean(:q1),
#             q3 = Statistics.mean(:q3)
#         )
#         @transform(:fitness_evaluations = population_size * :generation)
#     end
# end

end