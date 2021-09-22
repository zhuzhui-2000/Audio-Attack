import json

with open("SpeechCommands/speech_commands_v0.02/testing_list.txt", "r") as f:  # 打开文件
    data_test = f.readlines()  # 读取文件
    print(data_test[0])
f.close()

with open("SpeechCommands/speech_commands_v0.02/validation_list.txt", "r") as f:  # 打开文件
    data_val = f.readlines()  # 读取文件
    
f.close()







'''
for data_line in data_test:
    new_dict = {}
    data_line = data_line.replace('\n', '')
    tran_wav = data_line.split("/")
    new_dict = {}

    temp_transcript_path = transcript_path + data_line[:-3] + "txt"
    print(temp_transcript_path)  
    with open(temp_transcript_path, "w") as f:  # 打开文件
        f.write(tran_wav[0].upper())   # 读取文件

    #new_dict['wav_path'] = 100
    #print(tran_wav)

for data_line in data_val:
    new_dict = {}
    data_line = data_line.replace('\n', '')
    tran_wav = data_line.split("/")
    new_dict = {}

    temp_transcript_path = transcript_path + data_line[:-3] + "txt"
    print(temp_transcript_path)  
    with open(temp_transcript_path, "w") as f:  # 打开文件
        f.write(tran_wav[0].upper())   # 读取文件

    #new_dict['wav_path'] = 100
    #print(tran_wav)

'''
root_path = {'root_path': "/home/mmc-2018012484/SpeechCommands/speech_commands_v0.02/"}
test_json = []
val_json = []
transcript_path = "SpeechCommands/speech_commands_v0.02/"

for data_line in data_test:

    data_line = data_line.replace('\n', '')
    tran_wav = data_line.split("/")
    new_dict = {}

    temp_transcript_path = data_line[:-3] + "txt"

    new_dict['wav_path'] = data_line
    new_dict['transcript_path'] = temp_transcript_path
    test_json.append(new_dict)

    #new_dict['wav_path'] = 100
    #print(tran_wav)

root_path['samples'] = test_json
with open("deepspeech.pytorch/data/command_test.json","w") as f:
    json.dump(root_path,f)

for data_line in data_val:
    new_dict = {}
    data_line = data_line.replace('\n', '')
    tran_wav = data_line.split("/")
    new_dict = {}

    temp_transcript_path = data_line[:-3] + "txt"

    new_dict['wav_path'] = data_line
    new_dict['transcript_path'] = temp_transcript_path
    val_json.append(new_dict)

    #new_dict['wav_path'] = 100
    #print(tran_wav)

root_path['samples'] = val_json
with open("deepspeech.pytorch/data/command_val.json","w") as f:
    json.dump(root_path,f)
    




