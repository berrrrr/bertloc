import datetime
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

PATH = 'model/bertloc_model.pth'

# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 디바이스 설정
# device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

"""
# 데이터로드
"""

## train set
train_full = pd.read_csv('data/sample_data.csv', index_col=0)
columns = ['label', 'placename1', 'placename2', 'cplacename1', 'cplacename2',
           'catefullpath1', 'catefullpath2', 'address1', 'address2', 'phonelist1',
           'phonelist2', 'namedistance', 'wtmx1', 'wtmx2', 'wtmy1', 'wtmy2',
           'full_placename1', 'full_placename2', 'cname_features', 'cate_features',
           'phone_features', 'addr_features', 'distance', 'addr0', 'addr1',
           'addr2', 'addr3', 'addr4', 'addr5', 'phone0', 'phone1', 'phone2',
           'phone3', 'cate0', 'cate1', 'cate2', 'cate3', 'cate4', 'cname0',
           'cname1', 'cname2', 'truth_logit', 'false_logit', 'truth_softmax',
           'false_softmax']
print(train_full.shape)
train = train_full.loc[:, columns]
train = train.fillna(value="")

"""
# **전처리 - 훈련셋**
"""

# 라벨 추출
labels = train['label'].values
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


def bert_sentence_pair_preprocessing(dataset: pd.DataFrame, tokenizer: BertTokenizer, max_sequence_length=64):
    max_bert_input_length = 70

    dataset_input_ids = torch.empty((len(dataset), max_bert_input_length), dtype=torch.long)
    dataset_token_type_ids = torch.empty((len(dataset), max_bert_input_length), dtype=torch.long)
    dataset_attention_masks = torch.empty((len(dataset), max_bert_input_length), dtype=torch.long)
    dataset_lengths = torch.empty((len(dataset), 1), dtype=torch.long)
    dataset_labels = torch.empty((len(dataset), 1), dtype=torch.long)
    dataset_other_type_ids = torch.empty((len(dataset), 18), dtype=torch.long)
    # dataset_input_tensors = torch.empty(len(dataset), 4, max_bert_input_length, dtype=torch.float)

    for idx, data in dataset.iterrows():
        tokens = []
        input_type_ids = []

        # other type 전처리
        other_type_ids = []
        other_type_ids.append(data['addr0'])
        other_type_ids.append(data['addr1'])
        other_type_ids.append(data['addr2'])
        other_type_ids.append(data['addr3'])
        other_type_ids.append(data['addr4'])
        other_type_ids.append(data['addr5'])
        other_type_ids.append(data['phone0'])
        other_type_ids.append(data['phone1'])
        other_type_ids.append(data['phone2'])
        other_type_ids.append(data['phone3'])
        other_type_ids.append(data['cate0'])
        other_type_ids.append(data['cate1'])
        other_type_ids.append(data['cate2'])
        other_type_ids.append(data['cate3'])
        other_type_ids.append(data['cate4'])
        other_type_ids.append(data['cname0'])
        other_type_ids.append(data['cname1'])
        other_type_ids.append(data['cname2'])

        dataset_other_type_ids[idx] = torch.tensor(other_type_ids, dtype=torch.long)

        sentence_1_tokenized, sentence_2_tokenized = tokenizer.tokenize(data['full_placename1']), tokenizer.tokenize(data['full_placename2'])

        tokens.append("[CLS]")
        input_type_ids.append(0)

        for token in sentence_1_tokenized:
            tokens.append(token)
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in sentence_2_tokenized:
            tokens.append(token)
            input_type_ids.append(1)

        tokens.append("[SEP]")
        input_type_ids.append(1)

        # 전처리한 token 바탕으로 인덱스값 얻음
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # attention mask 전처리
        attention_masks = [1] * len(input_ids)

        # input_ids length 저장
        dataset_lengths[idx] = torch.tensor(len(input_ids), dtype=torch.long)

        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)

        dataset_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        dataset_token_type_ids[idx] = torch.tensor(input_type_ids, dtype=torch.long)
        dataset_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)

        dataset_labels[idx] = torch.tensor(data['label'], dtype=torch.long)

    return dataset_input_ids, dataset_token_type_ids, dataset_attention_masks, dataset_other_type_ids, dataset_lengths, dataset_labels


input_ids_eval, token_type_ids_eval, attention_masks_eval, other_type_ids_eval, input_length_eval, correct_labels_eval = bert_sentence_pair_preprocessing(train, tokenizer)

print("")

## train : test : validation = 6 : 2 : 2

# input_ids 훈련셋과 테스트셋 분리
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids_eval,
                                                                        correct_labels_eval,
                                                                        random_state=1,
                                                                        test_size=0.2)

# input_ids 훈련셋을 다시 훈련셋과 검증셋으로 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_inputs,
                                                                                    train_labels,
                                                                                    random_state=1,
                                                                                    test_size=0.25)

# attention_mask 훈련셋과 검증셋으로 분리
train_masks, test_masks, train_types, test_types = train_test_split(attention_masks_eval,
                                                                    token_type_ids_eval,
                                                                    random_state=1,
                                                                    test_size=0.2)

# attention_mask 훈련셋을 다시 훈련셋과 검증셋으로 분리
train_masks, validation_masks, train_types, validation_types = train_test_split(train_masks,
                                                                                train_types,
                                                                                random_state=1,
                                                                                test_size=0.25)
# other_ids 훈련셋과 검증셋으로 분리
train_others, test_others = train_test_split(other_type_ids_eval,
                                             random_state=1,
                                             test_size=0.2)

# other_ids 훈련셋을 다시 훈련셋과 검증셋으로 분리
train_others, validation_others = train_test_split(train_others,
                                                   random_state=1,
                                                   test_size=0.25)

# input_length 훈련셋과 검증셋으로 분리
train_lengths, test_lengths = train_test_split(input_length_eval, random_state=1, test_size=0.2)

# input_length 훈련셋을 다시 훈련셋과 검증셋으로 분리
train_lengths, validation_lengths = train_test_split(train_lengths, random_state=1, test_size=0.25)

# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_types = torch.tensor(train_types)
train_others = torch.tensor(train_others)
train_lengths = torch.tensor(train_lengths)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_types = torch.tensor(validation_types)
validation_others = torch.tensor(validation_others)
validation_lengths = torch.tensor(validation_lengths)
test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)
test_types = torch.tensor(test_types)
test_others = torch.tensor(test_others)
test_lengths = torch.tensor(test_lengths)

# 배치 사이즈
batch_size = 32

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_types, train_others, train_lengths, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_types, validation_others,
                                validation_lengths, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_types, test_others, test_lengths, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

"""
# 모델 선언
"""


# 분류를 위한 biLSTM 모델 설정

class CustomBertLocModel(nn.Module):

    def __init__(self, device, lstm_hidden_size=None):
        """
        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param lstm_hidden_size: int
        """

        super(CustomBertLocModel, self).__init__()

        self.bert_config = 'bert-base-multilingual-cased'
        self.bert = BertModel.from_pretrained(self.bert_config).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = 2
        self.dropout_rate = 0.5
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True).to(device)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, self.n_class, bias=True).to(device)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device
        self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 64, bias=True).to(device)

        self.conv1 = nn.Conv1d(2, 32, kernel_size=3).to(device)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3).to(device)

        self.fc2 = nn.Linear(18, 64).to(device)
        self.fc3 = nn.Linear(384, 2).to(device)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                input_lengths=None,
                other_type_ids=None):

        # bert layer
        encoded_layers, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids,
                                                  output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0, 2).to(device)

        # lstm layer
        enc_hiddens, (last_hidden, last_cell) = self.lstm(
            pack_padded_sequence(encoded_layers, input_lengths.squeeze(), enforce_sorted=False))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        output_hidden = self.fc1(output_hidden)

        # cnn layer
        other_type_ids = self.fc2(other_type_ids.type_as(output_hidden))
        conv_input = torch.stack([output_hidden, other_type_ids], dim=1)
        conv_out = F.relu(F.max_pool1d(self.conv1(conv_input), 3))
        conv_out = F.relu(F.max_pool1d(self.conv2(conv_out), 3))
        conv_out = conv_out.view(-1, 384)
        conv_out = self.dropout(conv_out)

        # softmax
        pre_softmax = self.fc3(conv_out.to(device))

        return F.log_softmax(pre_softmax, dim=1)


model = CustomBertLocModel(device)


def learning_rate_dacay(optimizer, epoch):
    # learning rate /=2 each two epoch after 10 epochs
    lr = 1e-5 * (0.1 ** max(0, (epoch - 9) // 2))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# 에폭수
epochs = 3

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs

# 학습률을 조금씩 감소시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    pred_flat = preds.squeeze()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# confusion matrix
def confusion(prediction, truth):
    pred_flat = prediction.squeeze()
    truth_flat = truth.flatten()

    confusion_vector = pred_flat / truth_flat

    true_positives = np.sum(confusion_vector == 1).item()
    false_positives = np.sum(confusion_vector == float('inf')).item()
    true_negatives = np.sum(np.isnan(confusion_vector)).item()
    false_negatives = np.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()
    learning_rate_dacay(optimizer, epoch_i)

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_input_types, b_input_others, b_input_lengths, b_labels = batch

        optimizer.zero_grad()

        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=b_input_types,
                        attention_mask=b_input_mask,
                        other_type_ids=b_input_others,
                        input_lengths=b_input_lengths)

        targets = torch.squeeze(b_labels)
        loss = F.cross_entropy(outputs, targets)

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_i, step * len(b_input_ids), len(train_dataloader.dataset), 100. * step / len(train_dataloader),
                loss.item()))

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    test_loss = 0
    correct = 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_input_types, b_input_others, b_input_lengths, b_labels = batch
        b_labels = torch.tensor(b_labels).to(device)

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=b_input_types,
                            attention_mask=b_input_mask,
                            other_type_ids=b_input_others,
                            input_lengths=b_input_lengths)
            targets = torch.squeeze(b_labels)

        test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()
        # 로스 구함
        logits = outputs.max(1, keepdim=True)[1]
        correct += logits.eq(targets.view_as(logits)).sum().item()

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        tmp_tp, tmp_fp, tmp_tn, tmp_fn = confusion(logits, label_ids)
        tp += tmp_tp
        fp += tmp_fp
        tn += tmp_tn
        fn += tmp_fn

        nb_eval_steps += 1

        test_loss /= len(test_dataloader.dataset)
        test_accuracy = 100. * correct / len(test_dataloader.dataset)

    print("  Accuracy: {0:.3f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("precesion : ", precision)
        print("recall : ", recall)
        print("f1 : ", f1)
    except:
        pass

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epochs, test_loss, test_accuracy))

print("")
print("Training complete!")

# 모델저장
torch.save(model.state_dict(), PATH)

"""
# **테스트셋 평가**
"""

# 시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
tp, fp, tn, fn = 0, 0, 0, 0
test_loss = 0
correct = 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_input_types, b_input_others, b_input_lengths, b_labels = batch

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행

        outputs = model(b_input_ids,
                        token_type_ids=b_input_types,
                        attention_mask=b_input_mask,
                        other_type_ids=b_input_others,
                        input_lengths=b_input_lengths)
        targets = torch.tensor(b_labels).to(device)

    # torch.save(model, PATH)

    # 로스 구함
    logits = outputs.max(1, keepdim=True)[1]
    correct += logits.eq(targets.view_as(logits)).sum().item()

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy

    tmp_tp, tmp_fp, tmp_tn, tmp_fn = confusion(logits, label_ids)
    tp += tmp_tp
    fp += tmp_fp
    tn += tmp_tn
    fn += tmp_fn

    nb_eval_steps += 1

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * correct / len(test_dataloader.dataset)

print("")
print("Accuracy: {0:.3f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))
print("tp: ", tp)
print("fp: ", fp)
print("tn: ", tn)
print("fn: ", fn)

try:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("precesion : ", precision)
    print("recall : ", recall)
    print("f1 : ", f1)
except:
    pass

print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
    epochs, test_loss, test_accuracy))
