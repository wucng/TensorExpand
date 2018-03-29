# -*- coding:utf-8 -*-

'''
解析配置文件，配置文件格式参考：darknet cfg文件格式
参考：https://pjreddie.com/darknet/
https://pjreddie.com/darknet/train-cifar/
'''

class Config_file_analyze(object):
    def __init__(self,config='cifar.cfg'):
        self.config=config
        self.org=['[net]','[convolutional]','[maxpool]','[connected]','[dropout]','[avgpool]','[softmax]','[cost]']
        self.network = []
        self.net = {}
        self.analyze_config_file()

    def add_list(self,cont,network):
        num=0
        while True:
            cont_=cont[1:-1] + str(num)
            if cont_ not in network:
                network.append(cont_)
                break
            else:
                num+=1
        return cont_


    def read_each_part(self,network, num, date, cont, net, org):
        # network.append(cont[1:-1] + str(num))
        cont_=self.add_list(cont,network)
        conv = {}
        for i in range(num + 1, len(date)):
            if date[i] == '': continue
            if date[i] in org: break
            if date[i][0]=='#':continue  # 跳过被注释的语句
            da = date[i].split('=')
            # conv[da[0].strip()] = da[1].strip() # 只能去除开头与末尾的空格
            conv[da[0].replace(' ', '')] = da[1].replace(' ', '') # 去除所有空格

        net[cont_] = conv
        conv=None

    def analyze_config_file(self):
        with open(self.config, 'r') as fp:
            # cont = cont.strip('\n').strip('\r').strip(' ')  # 去除掉换行符、空格
            # data=fp.readlines()
            # data = ''.join(data).strip('\n') # 删除换行符
            date = fp.read().splitlines()  # 删除换行符
            # data.remove('\n') # 一次只能删除一个
            for num, cont in enumerate(date):
                if cont == '': continue
                if cont in self.org:
                    self.read_each_part(self.network, num, date, cont, self.net, self.org)
                else:
                    continue


if __name__=='__main__':
    conf=Config_file_analyze()
    network=conf.network
    net=conf.net
    pass
