# Bogumił Kamiński, 2019-03-25

# Start Julia run with the following command:
#     julia -p auto
#
# .int file format specification:
#     * Each image is encoded by 1+28^2 bytes.
#     * Images are stored consecutively.
#     * First byte for each image is its class from 0 to 9
#     * Then each of remaining 28^2 has value from 0 to 255
#       and encodes one entry in a 28x28 image

using Statistics
using Distributed

try
    using PyPlot
    @info "PyPlot loaded"
catch
    @warn "PyPlot missing"
end

cd("/Users/maciej/git/inne/julia-misc/szkolenie/UEP-Workshop-20190405-master/6. MNIST/")

@everywhere const TRAIN = (class=UInt8[], image=Matrix{Int32}[])
@everywhere const TEST = (class=UInt8[], image=Matrix{Int32}[])

@everywhere for (filename, data) in [("fasionmnist_train.int", TRAIN),
                                     ("fasionmnist_test.int", TEST)]
    open(filename) do f
        while !eof(f)
            c = read(f, UInt8)
            v = read(f, 28^2)
            push!(data.class, c)
            push!(data.image, reshape(v, 28, 28))
        end
    end
end

@everywhere function distance(a, b)
    d = 0
    @simd for i in 1:length(a)
        @inbounds d += (a[i] - b[i]) ^ 2
    end
    d
end

@everywhere function knnacc(i)
    dist = distance.(Ref(TEST.image[i]), TRAIN.image)
    knn_loc = partialsortperm(dist, 1:20)
    TRAIN.class[knn_loc]
end

@time neivec = pmap(knnacc, axes(TEST.class, 1))
const NEIGHBOURS = reduce(hcat, neivec)

function acc(classes, testclass)
    v = zeros(Int, 10)
    for c in classes
        v[c + 1] += 1
    end
    m = maximum(v)
    (v[testclass + 1] == m) / count(==(m), v)
end

function evalk(k)
    mean(axes(NEIGHBOURS, 2)) do i
        classes = view(NEIGHBOURS, 1:k, i)
        testclass = TEST.class[i]
        acc(classes, testclass)
    end
end

wynik = evalk.(axes(NEIGHBOURS, 1))

# argmax -- which.max
# argmin -- which.min
argmax(wynik)
argmin(wynik)

plot(1:20, wynik)
