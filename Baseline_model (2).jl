## import Pkg 
import Pkg;

## using packages
using CSV;
using DelimitedFiles, Statistics, Random
import Pkg; Pkg.add("DataStructures"); using DataStructures;
using Base.Iterators: take, drop, cycle, Stateful
using IterTools: ncycle, takenth, takewhile
using Knet
using Knet: minibatch
import Pkg; Pkg.add("Tables")
using Tables

## read data from source
function read_data(data_source)
    data = CSV.File(data_source) |> Tables.matrix
    return data
end

# read word_all.txt: dictionary for hate words.
function read_txt_data(data_source)
    f = open(data_source)
    lines = readlines(f)
    return lines
end

## eliminate irrelated columns
function read_data_process(data)
    data = data[:, 2:3]
    return data
end

## read Sentiment Analysis Data
SA_data = read_data("train_E6oV3lV.csv")

## data processing
SA_data = read_data_process(SA_data)

hate_speech = read_txt_data("word_all.txt")

function target_ratio(data)
    sum(data[:,1])/size(data,1)
end

SA_rt= target_ratio(SA_data)
SA_rt

function train_test_split(df, test_split_size=0.2, dev_split_size=0.1, shuffle::Bool=false)
    if shuffle == true
       df = df[shuffle(1:end), :]
    end
    
    df_test = df[1:convert(Int,round(size(df,1)*test_split_size)),:]
    df_dev = df[convert(Int,round(size(df,1)*test_split_size))+1:(convert(Int,round(size(df,1)*test_split_size))+1)+convert(Int,round(size(df,1)*dev_split_size)) ,:]
    df_train = df[(convert(Int,round(size(df,1)*test_split_size))+1)+convert(Int,round(size(df,1)*dev_split_size)):end, :]
    
    return df_test, df_dev, df_train
end

SA_test,SA_dev,SA_train = train_test_split(SA_data)

function word_frequency(dataset, hate_speech)
    word_dict = Dict()
    for i in 1:size(dataset,1) 
        count=0
        for word in split(dataset[i])
            if word in hate_speech
                count +=1
            end
        end
        data = dataset[i]
        word_dict[data] = count    
    end
    return word_dict
end

SA_train_filtered = SA_train[(SA_train[:,1] .== 1),:]

batch_size=50

SA_train_filtered_minibatch = minibatch(SA_train_filtered[:,2],SA_train_filtered[:,1],batch_size)

size(SA_train_filtered,1)

n = 0; next_1 = iterate(SA_train_filtered_minibatch)
while next_1 != nothing
    ((_x,_y), state) = next_1
    n = n+sum(values(word_frequency(_x,hate_speech)))
    global next_1 = iterate(SA_train_filtered_minibatch,state)
end

avg_hate_speech_count  = n/size(SA_train_filtered,1)

## we found avg hate speech count for tweet which labelled as hate speech in training. It is 0.5. If tweet in test data set has 
## hate word at least 0.5 then we estimate this test tweet as hate speech

test_dimension = size(SA_test,1)
y_hat = zeros(test_dimension,1)

function prediction(test_set)
    for i in 1:size(test_set,1)
        count=0
        for word in split(test_set[i,2])
            if word in hate_speech
                count +=1
            end
        end
        if count >= avg_hate_speech_count
                y_hat[i] = 1
        end 
    end
 return y_hat
end

split(SA_test[1,2])

size(prediction(SA_test),1)

size(y_hat)

function accuracy(y_actual,y_pred)
    count=0
    for i in 1:size(y_actual,1)
        if y_actual[i] == y_pred[i]
            count +=1
        end
    end
    return count/size(y_actual,1)
end

accuracy(SA_test[:,1],y_hat)

function precision(y_actual,y_pred)
    count=0
    for i in 1:size(y_actual,1)
        if y_actual[i] == y_pred[i] && y_actual[i] == 1
            count +=1
        end
    end
    return count/sum(y_hat[(y_hat[:,1] .== 1),:])
end

p=precision(SA_test[:,1],y_hat)

function recall(y_actual,y_pred)
    count=0
    for i in 1:size(y_actual,1)
        if y_actual[i] == y_pred[i] && y_actual[i] == 1
            count +=1
        end
    end
    return count/sum(y_actual[(y_actual[:,1] .== 1),:])
end

r=recall(SA_test[:,1],y_hat)

##2*((precision*recall)/(precision+recall))

function f1(precision,recall)
    return 2*((precision*recall)/(precision+recall))
end

f1_score=f1(p,r)
