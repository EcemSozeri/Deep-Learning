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

SA_data = read_data_process(SA_data)

hate_speech = read_txt_data("word_all.txt")

SA_data[1:end,2]

sizeof(SA_data)

size(SA_data,1)

split(SA_data[1000,2]," ")

## remove emoji 

function remove_emoji(string)
    for i in 1:size(string,1)
      string[i,2]= replace(string[i,2], !isascii => " ") 
    end
    return string
end

## clear space if there is more than one in succession
function remove_spaces(text_string)
    for i in 1:size(text_string,1)
        text_string[i,2] = replace(text_string[i,2], r"(\s+)" =>' ' )
    end
    return text_string
end

## clear mentions
function remove_mention(text_string)
    for i in 1:size(text_string,1)
      text_string[i,2] = replace(text_string[i,2], r"@[\w\-]+" =>' ' ) 
    end
    return text_string
end

# remove hashtags
function remove_hashtag(text_string)
    for i in 1:size(text_string,1)
        text_string[i,2] = replace(text_string[i,2], r"#[\w\-]+" =>' ' )
    end
    return text_string
end

# remove "..."
function remove_ellipsis(text_string)
    for i in 1:size(text_string,1)
        text_string[i,2]  = replace(text_string[i,2] , "..." =>' ' )
    end 
    return text_string
end

# remove punctuation mark
function remove_punctuation_mark(text_string)
    for i in 1:size(text_string,1)
        text_string[i,2] = replace(text_string[i,2], [',',';','.','!','?','|',':','/','{','}','ð','#','[',']','@','\'','(',')','+',',','+','~','_','$','-'] => "")
    end 
    return text_string
end 
                ##&amp,

replace(string, r"(\s+)" =>' ' )

remove_punctuation_mark(SA_data)

## remove space if string starts with space
function remove_space_starts_with(text_string)
    for i in 1:size(text_string,1)
        if startswith(text_string[i,2]," ",)
            text_string[i,2]=lstrip(text_string[i,2])
        end
    end 
    return text_string
end

## all data pre process
## data processing
function all_data_process(SA_data)
    SA_data = remove_emoji(SA_data)
    SA_data = remove_mention(SA_data)
    SA_data = remove_hashtag(SA_data)
    SA_data = remove_ellipsis(SA_data)
    SA_data = remove_punctuation_mark(SA_data)
    SA_data = remove_spaces(SA_data)
    SA_data = remove_space_starts_with(SA_data)
    return SA_data
end

all_data_process(SA_data)

function tokenize(string)
    tokens=split(string," ")
    return tokens
end

tokenize(SA_data[1102,2])

'^[+-]?[0-9]+\.?[0-9]*$'

function get_dict()
    dict = Dict()
    alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','-',',',';','.','!','?',':','\'','\\','/','\\','|','_','@','#','$','%','^','&','*','~','`','+','-','=','<','>','(',')','[',']','{','}']
    for (i,c) in enumerate(alphabet)
        dict[c] = i
    end
    return dict
end

dict=get_dict()

function strToIndexs2(text,length_1 = 300)
    text = lowercase(text)
    m = length(text)
    n = min(m, length_1)
    str2idx = zeros(Int64,length_1)
    dict = get_dict()
    for i in 1:n
        c = text[i]
        if c in keys(dict)
            str2idx[i] = dict[c]
        end
    end
    return str2idx
end

function create_vocab(tweets, vocab_size=0)
    total_words, unique_words = 0, 0
    word_freqs = Dict()
    for i in 1:size(SA_data,1)
     tweet = SA_data[i,2]
     content = tokenize(tweet)
     for word in content
            try
                word_freqs[word] += 1
            catch
                unique_words += 1
                word_freqs[word] = 1
            end
            total_words += 1
     end
     end
     sorted_word_frequency=sort(collect(word_freqs),by=x->x[2],rev=true)
     if vocab_size <= 0
        vocab_size = 0
        for (word,freq) in sorted_word_freqs
            if freq >= 1
               vocab_size += 1
            end
        end
    end
    vocab = Dict("<pad>"=> 0, "<unk>"=> 1, "<word>"=> 2, "<no_word>"=> 3)
    vcb_len = length(vocab)
    index = vcb_len
    vocab_size-vcb_len
    for (word, _) in sorted_word_freqs[1:vocab_size - vcb_len]
        vocab[word] = index
        index += 1
    end 
    return vocab
end

vocab= create_vocab(SA_data)

function get_indices(tweets, vocab, hate_speech)
     data_x, char_x, ruling_embedding, category_embedding = [], [], [], []
     unk_hit, total = 0., 0.
     for i in 1:size(tweets,1)
        tweet = tweets[i]
        # tweet = tweets[i]
        
        indices_char = []
        indices_char = strToIndexs2(tweet)
        # tweet = tweets[i]
        # print(tweet)
        content = tokenize(tweet)
        n = length(content)
        t = false
        indices = []
        category_indeics = []  # 脏词为0，错词为1，其他为2
        for word in content
            print("----- yeni word------")
            print(word)
            print("---------------------")
            print("\n")
            for j in 1:n
                # if is_number(word):
                #     indices.append(vocab['<num>'])
                #     num_hit += 1
                word = content[j]
                if j < n-1
                    global word_2 = content[j]*" "*content[j+1]
                    print("----- word_2------")
                    print(word_2)
                    print("---------------------")
                    print("\n")
                end
                if j < n-2
                    global word_3 = content[j]*" "*content[j+1]*" "*content[j+2]
                    print("----- word_3------")
                    print(word_3)
                    print("---------------------")
                    print("\n")
                end
        
                if word in hate_speech || word_2 in hate_speech || word_3 in hate_speech  # 3-gram
                   t = true
                end
        
                if word in keys(vocab)
                    append!(indices,vocab[word])
                else
                    append!(indices,vocab["<unk>"])
                    unk_hit += 1
                end
        
                if word in hate_speech
                    append!(category_indeics,0)
                elseif  ('@' ∉ word) && (word ∉ [':', ',', "''", "''", '!', "'s", '?', "facebook", "n't","'re", "'", "'ve", "everytime"])
                    append!(category_indeics,1)
                else
                    append!(category_indeics,2)
                end
                total += 1
        
                if t
                    ruling = [2]*50   
                else
                    ruling = [3]*50
                end
                append!(ruling_embedding,ruling)    # It corresponds to category embedding in the paper
                append!(data_x,indices)
                append!(char_x,indices_char)
                append!(category_embedding,category_indeics)
            end
        end
    end
    return data_x, char_x, ruling_embedding, category_embedding
end

get_indices(SA_data[:,2], vocab, hate_speech)

SA_data[:,2]

total = 0
data_x, char_x, ruling_embedding, category_embedding = [], [], [], []
content = tokenize(tweet)
n = length(content)
t = false
indices = []
category_indeics = []  # 脏词为0，错词为1，其他为2
indices_char = []
indices_char = strToIndexs2(tweet)
for word in content
    print("----- yeni word------")
    print(word)
    print("---------------------")
    print("\n")
    for j in 1:n
        # if is_number(word):
        #     indices.append(vocab['<num>'])
        #     num_hit += 1
        word = content[j]
        if j < n-1
            global word_2 = content[j]*" "*content[j+1]
            print("----- word_2------")
            print(word_2)
            print("---------------------")
            print("\n")
        end
        if j < n-2
            global word_3 = content[j]*" "*content[j+1]*" "*content[j+2]
            print("----- word_3------")
            print(word_3)
            print("---------------------")
            print("\n")
        end
        
        if word in hate_speech || word_2 in hate_speech || word_3 in hate_speech  # 3-gram
                    t = true
        end
        
        if word in keys(vocab)
            append!(indices,vocab[word])
        else
            append!(indices,vocab["<unk>"])
            unk_hit += 1
        end
        
        if word in hate_speech
            append!(category_indeics,0)
        elseif  ('@' ∉ word) && (word ∉ [':', ',', "''", "''", '!', "'s", '?', "facebook", "n't","'re", "'", "'ve", "everytime"])
            ##append!(category_indeics,1)
        else
            append!(category_indeics,2)
        end
        total += 1
        
        if t
            ruling = [2]*50   
        else
            ruling = [3]*50
        end
        append!(ruling_embedding,ruling)    # It corresponds to category embedding in the paper
        append!(data_x,indices)
        append!(char_x,indices_char)
        append!(category_embedding,category_indeics)


    end
end

labels=SA_data[:,1]

function get_statistics(tweets, labels, hate_speech)
    a0, a1, a2, a3, a4 = 0, 0, 0, 0, 0
    b0, b1, b2, b3, b4 = 0, 0, 0, 0, 0
    hate, non_hate = 0, 0
    
    for i in 1:length(tweets)
        tweet = tweets[i]
        content = tokenize(tweet)
        label = labels[i]
        dirty_word_num = 0
        for word in content
            if word in word_list
                dirty_word_num += 1
            end
        end
        if label == 0
            non_hate += 1
            if dirty_word_num == 0
                a0 += 1
            elseif dirty_word_num == 1
                a1 += 1
            elseif dirty_word_num == 2
                a2 += 1
            elseif dirty_word_num == 3
                a3 += 1
            else
                a4 += 1
            end
        else
            hate += 1
            if dirty_word_num == 0
                b0 += 1
            elseif dirty_word_num == 1
                b1 += 1
            elseif dirty_word_num == 2
                b2 += 1
            elseif dirty_word_num == 3
                b3 += 1
            else
                b4 += 1
            end
        end  
    end
    print("hate, non_hate=", hate, non_hate)
    print("b0,b1,b2,b3,b4=", b0,b1,b2,b3,b4)
    print("a0,a1,a2,a3,a4=", a0,a1,a2,a3,a4)
end

function turn2(Y)
    for i in range(len(Y))
        if Y[i]==2
            Y[i] -= 1
        end
    end
    return Y
end

function read_dataset()
    data_train = CSV.File("hateval2019_en_train.csv";header=1, delim=",") |> Tables.matrix
    data_train=data_train[:,2:3]
    df_task['task_idx'] = [0]*len(df_task)

end

data_train = CSV.File("hateval2019_en_train.csv";header=1, delim=",") |> Tables.matrix
data_train=data_train[:,2:4]
data_train[:,3] = zeros(Int,size(data_train,1),1)
data_train

data_test = CSV.File("hateval2019_en_test.csv";header=1, delim=",") |> Tables.matrix
data_test=data_test[:,2:4]
data_test[:,3] = zeros(Int,size(data_test,1),1)
data_test

## Sentiment Analysis Data
df_sentiment = CSV.File("train_E6oV3lV.csv";header=1) |> Tables.matrix
df_sentiment = df_sentiment[:,2:4]
df_sentiment[:,3] = ones(Int,size(df_sentiment,1),1)
df_sentiment = df_sentiment[shuffle(1:end),:]
df_sentiment_train,df_sentiment_test = df_sentiment[1:2*trunc(Int,size(data_train,1)),:],df_sentiment[trunc(Int, 0.99*(size(df_sentiment,1))):end,:]

df_task_test = vcat([data_test, df_sentiment_test])
data_all = vcat([data_train, df_sentiment_train])
data_all = data_all[shuffle(1:end),:]

if  !vocab_path
    data = data_all
    tweets = vcat([data, df_task_test])
    tweets = tweets[:,1]
    vocab = create_vocab(tweets)
end

data_tokens, train_chars, ruling_embedding_train, category_embedding_train = get_indices(data_all[:,2], vocab, args.word_list_path)  
X_test_data, test_chars, ruling_embedding_test, category_embedding_test = get_indices(df_task_test[:,2], vocab, args.word_list_path)

Y = data_all[:,2] ##label
Y = turn2(Y)
y_test = df_task_test[:,2]



## devam et
y_test = np_utils.to_categorical(y_test)
dummy_y = np_utils.to_categorical(Y)
X_train_data, y_train = data_tokens, dummy_y

task_idx = np.array(list(data_all.task_idx), dtype='int32')
task_idx_train = np_utils.to_categorical(task_idx)
task_idx_test = np.array(list(df_task_test.task_idx), dtype='int32')
task_idx_test = np_utils.to_categorical(task_idx_test)

X_train_data = sequence.pad_sequences(X_train_data, maxlen=MAX_SEQUENCE_LENGTH)
X_test_data = sequence.pad_sequences(X_test_data, maxlen=MAX_SEQUENCE_LENGTH)
category_embedding_train = sequence.pad_sequences(category_embedding_train, maxlen=MAX_SEQUENCE_LENGTH)
category_embedding_test = sequence.pad_sequences(category_embedding_test, maxlen=MAX_SEQUENCE_LENGTH)

return X_train_data, X_test_data, y_train, y_test, np.array(train_chars), np.array(test_chars), task_idx_train, task_idx_test, np.array(ruling_embedding_train, dtype=np.float), \
np.array(ruling_embedding_test, dtype=np.float), category_embedding_train, category_embedding_test, vocab

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
