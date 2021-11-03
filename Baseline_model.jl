## import Pkg to 
import Pkg;

## using packages
using CSV;
using DelimitedFiles, Statistics, Random
import Pkg; Pkg.add("DataStructures"); using DataStructures;
using Knet: minibatch

## read data function
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

## read Sentiment Analysis Data
SA_data = read_data("train_E6oV3lV.csv")
SA_data

function read_data_process(data)
    data = data[:, 2:3]
    return data
end

SA_data = read_data_process(SA_data)

function target_ratio(data)
    a=counter(data[:,1])
    target_rate=0
    for key in keys(a)
        if key == 1
            target_rate = a[1]/sum(values(a))
        end
    end
    return target_rate
end

rt= target_ratio(SA_data)
rt

SA_data

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
        for word in split(dataset[i,2])
            if word in hate_speech
                count +=1
            end
        end
        data = dataset[i,2]
        word_dict[data] = count
        print(data)
        print(word_dict[data])     
    end
    return word_dict
end

hate_speech = read_txt_data("word_all.txt")

SA_train[1:15000,:]

i = word_frequency(SA_train,hate_speech)
i

keys(my_dict)

size(SA_train,1)

target_ratio(SA_train)

batch_size=100

dtrn = minibatch(SA_train[:,2],SA_train[:,1],batch_size)

minibatch_number = length(dtrn)

total=0
count=0
for (x,y) in dtrn
    total += sum(y)
end
print(total)
total/(minibatch_number*batch_size)

SA_test[:,1]

size(SA_test,1)

y_hat[1]

y_hat = zeros(6390,1)

for i in 1:size(y_hat,1)
    if mod(i,14) == 0
       y_hat[i] = 1
    end
end

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


