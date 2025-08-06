You can run our code through the following: python main.py --method DiHyper  --dname $dname  --second_name $second_name  

The following additional content is about how to run all parameter experiments. Please change the dataset path and parameters according to the actual situation.


import re
def get_namespace(namespace_str):
    pattern = r'(\w+)=([^\s,]+),'
    matches = re.findall(pattern, namespace_str) 
    namespace_dict = {}
    for key, value in matches:
        if value == 'None':
            namespace_dict[key] = None
        elif value == 'True':
            namespace_dict[key] = True
        elif value == 'False':
            namespace_dict[key] = False
        elif value.replace('.', '', 1).isdigit():
            namespace_dict[key] = float(value) if '.' in value else int(value)
        elif value.startswith("'") and value.endswith("'"):
            namespace_dict[key] = value.strip("'")
        else:
            namespace_dict[key] = value
    return namespace_dict



def get_max_accuracy_line(filename):
    max_accuracy = -1
    max_accuracy_line = ""
    
    with open(filename, 'r') as file:
        for line in file:
            try:
                match = line.strip().split('\t')
                test_acc = match[2]
                if test_acc:
                    accuracy_match = re.search(r'(\d+\.\d+)', test_acc)
                    if accuracy_match:
                        accuracy = float(accuracy_match.group())
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            max_accuracy_line = line
            except:
                pass
    return max_accuracy_line

def get_num(result_path):    
    results = []
    with open(result_path, 'r') as file:
        for line in file:
            results.append(line.strip())
    return len(results)

def get_results(result_path):    
    results = []
    with open(result_path, 'r') as file:
        for line in file:
            results.append(line.strip())
    return results


dname = 'WebKB'
second_name = 'texas'
method = 'DiHyper'
alpha = 1
gamma = 1
res_root1 = 'result'
raw_data_dir = f'data/original_data/{second_name}/'
data_dir = f'data_data_dir\{second_name}'
epochs = 500



for MLP_hidden in [64, 128, 256, 512]:
    for Classifier_hidden in [64, 128, 256]:
        for nconv in range(1,5):
            %run main.py \
            --method $method\
            --alpha $alpha  \
            --gamma $gamma  \
            --dname $dname  \
            --second_name $second_name  \
            --res_root $res_root1 \
            --connection True \
            --nconv $nconv  \
            --Classifier_num_layers 2 \
            --hidden $MLP_hidden \
            --Classifier_hidden $Classifier_hidden \
            --wd 0.005 \
            --epochs $epochs  \
            --runs 10   \
            --directed True  \
            --data_dir $data_dir \
            --raw_data_dir $raw_data_dir \
            --ablation full


dname = 'WebKB'
second_name = 'texas'
method = 'DiHyper'
alpha = 1
gamma = 1
res_root1 = 'result'
raw_data_dir = f'data/original_data/{second_name}/'
data_dir = f'data_data_dir\{second_name}'
epochs = 500

filename = f'{res_root1}/{second_name}.csv'
namespace_str = get_max_accuracy_line(filename)
namespace_dict = get_namespace(namespace_str)

nconv= namespace_dict['nconv']
MLP_hidden= namespace_dict['hidden'] 
Classifier_hidden= namespace_dict['Classifier_hidden'] 
alpha = namespace_dict['alpha']
gamma = namespace_dict['gamma']

for i_alpha in range(9,0,-1):
    alpha = i_alpha/10
    for i_gamma in range(9,0,-1):
        gamma = i_gamma/10
        %run main.py \
        --method $method\
        --alpha $alpha \
        --gamma $gamma  \
        --dname $dname  \
        --second_name $second_name  \
        --res_root $res_root1 \
        --connection True \
        --nconv $nconv  \
        --Classifier_num_layers 2 \
        --hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd 0.005 \
        --epochs $epochs  \
        --runs 10   \
        --directed True  \
        --data_dir $data_dir \
        --raw_data_dir $raw_data_dir \
        --ablation full  

