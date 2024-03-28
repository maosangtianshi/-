from io import open
# 字符规范化
import unicodedata
# 正则表达式
import re
# 随机生成数据
import random
# 用于中文分词
import jieba
# 构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定的优化方法工具包
from torch import optim
# 使用sys.stdout.encoding
import sys
print(torch.cuda.is_available())
# 设备选为gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 时间计算
import time
import math

# 调用训练函数并打印日志和制图
import matplotlib.pyplot as plt

# 算模型得分
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 数据预处理:
# 将指定语言中词汇映射为数值
# #启示标志
SOS_token = 0
# # 结束标志
EOS_token = 1


class LangE:
    # 语言名字方法
    def __init__(self, name):
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典，0、1分别对应SOS和EOS
        self.index2word = {0: "SOS", 1: "EOS"}
        # 将数对词索引设为2
        self.n_words = 2

    # 添加句子方法
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # 添加词汇方法
    def addWord(self, word):
        # 不在就添加
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            # 索引加一
            self.n_words += 1

class LangC:
    # 语言名字方法
    def __init__(self, name):
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典，0、1分别对应SOS和EOS
        self.index2word = {0: "SOS", 1: "EOS"}
        # 将数对词索引设为2
        self.n_words = 2

    # 添加句子方法
    def addSentence(self, sentence):
        for word in jieba.lcut(sentence):
            self.addWord(word)

    # 添加词汇方法
    def addWord(self, word):
        # 不在就添加
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            # 索引加一
            self.n_words += 1


''' 字符规范化 '''

# 字符串规范化，s为输入的字符串
def normalizeString(s):
    # 如果字符串中包含中文字符，则保留原始字符串
    if re.search("[\u4e00-\u9fa5]", s):
        return s.strip()
    # 否则进行规范化处理
    else:
        # 变为小写并除去两侧空白符
        s = s.lower().strip()
        # 在.。！!？?前加一个空格
        s = re.sub(r"([.。！!？?])", r" \1", s)
        # 使用正则表达式将字符串中不是大小写字母和非正常标点替换为空格
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


# 读取语言函数，lang1为源语言名字，lang2为目标语言名字
def readLangs(lang1, lang2):
    # 读取每一行
    lines = open("train.txt", encoding='utf-8').read().strip().split('\n')
    # 对每一行进行标准化处理,并以\t划分，得到语言对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = LangE(lang1)
    output_lang = LangC(lang2)
    return input_lang, output_lang, pairs

'''过滤处符合要求的语言对'''

# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 20
# 选着带有指定前缀的语句作为训练数据
# eng_prefixes = ("i am", "i m", "he is", "he s", "she is", "she s", "you are","you re", "we are", "we re", "they are", "ther re")

# 过滤函数
# 传入p为语言对，p[0]为英文句子，p[1]为法文句子，长度均要小于设定最长长度
def filterPair(p):
    # return len(p[0].split(' ')) < MAX_LENGTH and len(jieba.lcut(p[1])) < MAX_LENGTH and p[0].startswith(eng_prefixes)
    return len(p[0].split(' ')) < MAX_LENGTH and len(jieba.lcut(p[1])) < MAX_LENGTH
# 对多个语言对过滤
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2):
    # 先通过readLangs获得input_lang,output_lang,pairs列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 过滤
    pairs = filterPairs(pairs)
    # 遍历
    for pair in pairs:
        # 使用addSentence方法进行数值映射
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

# 进行数据处理
input_lang, output_lang, pairs = prepareData("en", "cn")



'''张量转换'''

# 将语言转化为模型输入需要的张量
def tensorFromSentenceE(lang, sentence):
    # lang为Lang的实例化对象，sentence为预转换句子
    # 遍历句子，获取每个词汇对应索引，装为列表
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 加入句子结束标志'1'
    indexes.append(EOS_token)
    # 将其使用torch.tensor封装为张量，改变形状为nx1
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentenceC(lang, sentence):
    # lang为Lang的实例化对象，sentence为预转换句子
    # 遍历句子，获取每个词汇对应索引，装为列表
    indexes = [lang.word2index[word] for word in jieba.lcut(sentence)]
    # 加入句子结束标志'1'
    indexes.append(EOS_token)
    # 将其使用torch.tensor封装为张量，改变形状为nx1
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    # 将语言对转换为张量对
    input_tensor = tensorFromSentenceE(input_lang, pair[0])
    target_tensor = tensorFromSentenceC(output_lang, pair[1])
    # 返回二者组成的元组
    return (input_tensor, target_tensor)


''' GRU编码器 '''
# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size代表编码器的输入尺寸即源语言词表大小
        # hidden_size代表GRU的隐层节点数，也代表词嵌入维度，是GRU的输入尺寸
        super(EncoderRNN, self).__init__()  # 继承 nn.Module 类
        # 传入hidden_size
        self.hidden_size = hidden_size
        # 实例化nn中预定义的Embedding层，参数为input_size和hidden_size
        # hidden_size即为词嵌入维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 实例化nn中预定义的GRU层，参数为hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    # 前向逻辑函数
    def forward(self, input, hidden):
        # input代表源语言的Embedding层输入张量
        # hidden代表编码器层gru初始隐层张量
        # 将输入张量进行embedding操作，并使形状改为(1,1,-1),-1代表自动计算维度
        # 编码器每次只以一个词作为输入
        # torch中预定义的gru必须使用三维张量作为输入，拓展一个维度
        output = self.embedding(input).view(1, 1, -1)
        # 将embedding层的输出和传入的初始hidden作为gru的输入传入
        output, hidden = self.gru(output, hidden)
        # 获得最终gru输出的output和对应的隐层张量hidden
        return output, hidden

    # 初始化隐层张量函数
    def initHidden(self):
        # 1*1*self.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)

    ''' 解码器 '''

# 构建基于GRU和Attention的解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # hidden_size代表解码器中GRU的输入尺寸，是隐层节点数
        # output_size代表解码器输出尺寸，是目标语言的词表大小
        # dropout_p是dropout层的置零比率，默认0.1
        # max_length是矩阵最大长度
        super(AttnDecoderRNN, self).__init__()
        # 传入参数
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 实例化Embedding层
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        # Attention的QKV理论第一步的第一种:
        # Q，K纵轴拼接，做一次线性变换，再用softmax处理后与V做张量乘法
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 实例化一个线性层，用于规范输出尺寸
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 实例化一个nn.Dropout(self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
        # 实例化nn.GRU,输入和隐层尺寸都是self.hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 最后实例化gru后的线性层，即是解码器的输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)

    # 前向函数
    def forward(self, input, hidden, encoder_outputs):
        # 参数分别为源数据输入张量，初始的隐层张量，解码器的输出张量
        # 对输入张量进行Embedding层并扩展维度
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = embedded.to(device)
        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)
        # 将hidden移动到GPU上
        hidden = hidden.to(embedded.device)
        # 进行attention的权重计算
        # Q,K进行纵轴拼接，做一次线性变化，最后使用softmax处理结果
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 将 encoder_outputs 移动到 GPU 上
        encoder_outputs = encoder_outputs.to(embedded.device)
        # 将权重矩阵与V做乘法，二者都需要是三维张量
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 再次拼接，后通过取[0]降维
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # 对结果做线性变化并拓展维度
        output = self.attn_combine(output).unsqueeze(0)
        # 使用relu激活
        output = F.relu(output)
        # 将激活后结果作为gru的输入和hidden一起传入
        output, hidden = self.gru(output, hidden)
        # 最后降维结果并用softmax处理
        output = F.log_softmax(self.out(output[0]), dim=1)
        # 返回输出结果，隐藏层张量和权重张量
        return output, hidden, attn_weights

    # 初始化张量函数
    def initHidded(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)



    '''计时器'''
def timeSince(since):
    # since位训练开始时的时间
    # 获得当前时间
    now = time.time()
    # 获得时间差
    s = now - since
    # 秒化分,取整
    m = math.floor(s / 60)
    # 计算不足一分的秒数
    s -= m * 60
    # 返回耗时
    return '%dm %ds' % (m, s)

# 设定训练开始时间是10min之前
since = time.time() - 10 * 60


''' 训练函数 '''
# 构建训练函数
# 设置teacher_forcing比率
teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,max_length=MAX_LENGTH):
    # 参数分别为：源语言输入张量，目标语言输入张量，编码器和解码器实例化对象
    # 编码器和解码器优化方法，损失函数计算方法，句子最大长度
    encoder_hidden = encoder.initHidden()
    # 将编码器和解码器优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 根据源文本和目标文本张量获取对应长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化编码器输出张量，形状为max_lengthencoder.hidden_size的零张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 初始设置损失为0张量
    loss = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
    # 循环遍历输入张量索引
    for ei in range(input_length):
        # 根据索引从input_tensor中取出对应单词的张量表示，和初始化隐层张量一同传入encoder中
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 将输出的三维张量encoder_output用[0,0]降为二维存入到encoder_outputs
        # encoder_outputs每一行都是对应句子中每个单词通过解码器的输出
        encoder_outputs[ei] = encoder_output[0, 0]
    # 初始化解码器第一个输入，即起始符
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # 初始化编码器的隐层张量即编码器的隐层输出
    decoder_hidden = encoder_hidden
    # 根据随机数与teacher_forcing_ratio对比判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # 使用时
    if use_teacher_forcing:
        for di in range(target_length):
            # decoder_input,decoder_hidden,encoder_outputs即QKV
            # 传入编码器对象，获得decoder_output,decoder_hidden,decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,decoder_hidden, encoder_outputs)
            # 强制下一次编码器输入为target_tensor
            decoder_input = target_tensor[di]
            # 使用了teacher_forcing,只使用target_tensor[di]计算损失
            loss = loss + criterion(decoder_output, target_tensor[di])
    # 不使用时
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,decoder_hidden, encoder_outputs)
            # 从decoder_output取答案
            topv, topi = decoder_output.topk(1)
            # 仍使用decoder_output和target_tensor[di]计算损失
            loss = loss + criterion(decoder_output, target_tensor[di])
            # 遇到终止符停止循环
            if topi.squeeze().item() == EOS_token:
                break
            # 对topi降维并分离赋值给decoder_input以便下次运算
            decoder_input = topi.squeeze().detach()

    # 误差进行反向传播
    loss.backward()
    # 编码器和解码器优化器进行更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 返回平均损失
    return loss / target_length

'''迭代训练'''

# 迭代训练函数
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    # 参数为编码器，解码器，迭代步数，打印日志间隔，绘制曲线间隔，学习速率
    # 获得开始时间
    start = time.time()
    # 每个损失间隔的平均损失保存在列表，用于制图
    plot_losses = []

    # 每个打印日志间隔的总损失，初始为0
    print_loss_total = 0
    # 每个绘制损失间隔的总损失，初始为0
    plot_loss_total = 0
    # 使用预定义SGDzu作为优化器，将参数和学习速率传入
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 选择损失函数
    criterion = nn.NLLLoss()
    # 将模型和优化器移动到CUDA上
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()

    # 根据迭代步骤进行循环
    for iter in range(1, n_iters + 1):
        # 每次从语言对列表中随机取出一条作为训练语句
        training_pair = tensorsFromPair(random.choice(pairs))
        # 分别从training_pair中取出输入和目标张量
        input_tensor = training_pair[0].to(device)
        target_tensor = training_pair[1].to(device)
        # 通过train函数获得模型运行的损失
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # 损失累和
        print_loss_total += loss
        plot_loss_total += loss
        # 当迭代步骤到达日志打印间隙时
        if iter % print_every == 0:
            # 通过总损失除以间隙得到平均损失
            print_loss_avg = print_loss_total / print_every
            # 总损失归零
            print_loss_total = 0
            # 打印日志，日志内容：耗时，当前迭代步，进度百分比，当前平均损失
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, print_loss_avg))
        # 当迭代步达到损失绘制间隙
        if iter % plot_every == 0:
            # 总损失除以间隙
            plot_loss_avg = plot_loss_total / plot_every
            # 平均损失装入列表
            plot_losses.append(plot_loss_total)
            # 总损失归零
            plot_loss_total = 0
    # 绘制损失曲线
    plt.figure()
    plt.plot([loss.detach().cpu().numpy() for loss in plot_losses])
    # 保存路径
    plt.savefig("./s2s_loss.png")

    '''评估函数'''


# 构建模型评估函数
def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
    # 编码器，解码器，要评估句子，句子最大长度
    # 评估阶段不进行梯度计算
    # 开启评估模式
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # 输入句子张量表示
        input_tensor = tensorFromSentenceE(input_lang,sentence)
        # 获取句子长度
        input_length = input_tensor.size()[0]
        # 初始化编码器隐层张量
        encoder_hidden = encoder.initHidden()
        # 初始化编码器输出张量为max_length x encoder.hidden_size的零张量
        encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
        # 遍历输入张量的索引
        for ei in range(input_length):
            # 根据索引从input_tensor中取出对应单词张量，和初始化隐层张量一起传入encoder
            encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            # 将每次输出encoder_output降维存入encoder_outputs中
            encoder_outputs[ei] += encoder_output[0,0]
        # 初始化起始符(第一个输出）
        decoder_input = torch.tensor([[SOS_token]],device=device)
        # 初始化解码器隐层张量
        decoder_hidden = encoder_hidden
        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(max_length,max_length)
        # 循环解码
        for di in range(max_length):
            # 将decoder_output,decoder_hidden,dncoder_outputs传入解码器对象
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
            # 将attention结果存入attention中
            decoder_attentions[di] = decoder_attention.data
            # 从解码器输出中获得概率最高的值及其索引对象
            topv, topi = decoder_output.data.topk(1)
            # 从索引对象中获取它的值与结束标志及其索引对比
            if topi.item() == EOS_token:
                # 是则将结束标志装入decoded_words列表，结束
                decoded_words.append('<EOS>')
                break
            else:
                # 不是则找出它在输出语言index2word字典中对应的单词装入decoded_words
                decoded_words.append(output_lang.index2word[topi.item()])
            # 最后将预测的索引降维并分离赋值给decoder_input，以便下次预测
            decoder_input = topi.squeeze().detach()
        # 返回结果decoded_words，以及完整注意力张量，切掉没用到的部分
        return decoded_words,decoder_attentions[:di + 1]


def Get_test():
    # 读取每一行
    test_lines = open("test.txt", encoding='utf-8').read().strip().split('\n')
    # 对每一行进行标准化处理,并以\t划分，得到语言对
    test_pairs = [[normalizeString(s) for s in l.split('\t')] for l in test_lines]
    return test_pairs


def evaluate_dataset(encoder, decoder, test_pairs, num_samples=100):
    bleu_scores = []
    selected_pairs = random.sample(test_pairs, num_samples)

    for test_pair in selected_pairs:
        input_sentence = normalizeString(test_pair[0])
        reference_sentence = normalizeString(test_pair[1])


        while 1:
            # 检查输入句子是否包含字典中不存在的单词，如果是，则重新抽取一对新的样本
            if any(word not in input_lang.word2index for word in input_sentence.split()):
                test_pair = random.choice(test_pairs)
                input_sentence = normalizeString(test_pair[0])
                reference_sentence = normalizeString(test_pair[1])
                continue

            output_words, _ = evaluate(encoder, decoder, input_sentence)
            candidate = output_words
            reference = jieba.lcut(reference_sentence)
            print(candidate)
            print(reference)
            smoothing_function = SmoothingFunction().method5
            # 设置 weights 为(0.5,0.5)，考虑双元和四元 n-gram，不考虑单元和三元 n-gram。
            bleu_score = sentence_bleu([reference], candidate, weights=(0.5, 0.5),smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)
            break  # 如果成功计算 BLEU 分数，则跳出循环

    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return average_bleu_score

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words,hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size,output_lang.n_words,dropout_p=0.1).to(device)
n_iters = 50000
print_every = 2000

encoder1.load_state_dict(torch.load('encoder.pth'))
attn_decoder1.load_state_dict(torch.load('decoder.pth'))
print("加载模型成功")

# trainIters(encoder1,attn_decoder1,n_iters,print_every=print_every)
# # 保存编码器模型
# torch.save(encoder1.state_dict(), 'encoder.pth')
# # 保存解码器模型
# torch.save(attn_decoder1.state_dict(), 'decoder.pth')
# print("已保存模型")

test_pairs = Get_test()
# 使用模型计算 BLEU 分数
bleu_score = evaluate_dataset(encoder1, attn_decoder1, test_pairs)
print("BLEU Score:", bleu_score)
