import socket
import struct
import numpy as np
#CSV文件中个字段的位置
tcp_stream_name  = 0
timestamp        = 1
ip_len           = 2
payload_len      = 3
UAPRSF_tcp_flags = 4

#通过IP是否是公网IP矫正客户端与服务器
def ip_type(ip_str):
    # 将IPv4地址转换为整数
    ip_int = struct.unpack('!I', socket.inet_aton(ip_str))[0]
    
    # 私有IP地址范围
    private_ip_ranges = [
        {'start': struct.unpack('!I', socket.inet_aton('10.0.0.0'))[0], 'end': struct.unpack('!I', socket.inet_aton('10.255.255.255'))[0]},
        {'start': struct.unpack('!I', socket.inet_aton('172.16.0.0'))[0], 'end': struct.unpack('!I', socket.inet_aton('172.31.255.255'))[0]},
        {'start': struct.unpack('!I', socket.inet_aton('192.168.0.0'))[0], 'end': struct.unpack('!I', socket.inet_aton('192.168.255.255'))[0]},
    ]
    
    # 判断IP地址是否在私有IP范围内
    for range in private_ip_ranges:
        if range['start'] <= ip_int <= range['end']:
            return 'private'
    
    # 如果不在私有IP范围内，则为公网IP
    return 'public'
    
def load_meta_for_lstm_from_csv(fname,start,end):
    #从csv文件中读取数据
    tcp_meta = []
    with open(fname,'r',encoding='utf-8') as f:
        for line in list(f)[start:end]:
            if line[-2] == '!':
                tcp_meta += [line[:-2]]
                
    #将数据以tcp流为单位进行组合
    streams = dict()
    for item in tcp_meta:
        #获取元数据中的各属性
        item_split = item.split(',')
        #获取流名称
        stream_name = item_split[tcp_stream_name]
        srcsock,dstsock = stream_name.split('-')
        stream_name = min(srcsock,dstsock)+'-'+max(srcsock,dstsock)
        ip_v = float(item_split[ip_len])/1500
        payload_v = float(item_split[payload_len])/1500
        temp = [[
            (float(srcsock < dstsock)-0.5)*2, #方向先随便定义
            item_split[timestamp], #时间戳视为非数值，稍后单独处理
            float(ip_v if ip_v < 1.0 else 1 + np.log(ip_v)),
            float(payload_v if payload_v < 1.0 else 1 + np.log(payload_v)),
            float(item_split[UAPRSF_tcp_flags][0]=='1'),
            float(item_split[UAPRSF_tcp_flags][1]=='1'),
            float(item_split[UAPRSF_tcp_flags][2]=='1'),
            float(item_split[UAPRSF_tcp_flags][3]=='1'),
            float(item_split[UAPRSF_tcp_flags][4]=='1'),
            float(item_split[UAPRSF_tcp_flags][5]=='1'),
        ]]
        if stream_name in streams:
            streams[stream_name] += temp
            t1 = float(streams[stream_name][-2][1])
            t2 = float(streams[stream_name][-1][1])
            if t2 < t1 or t2 - t1 > 300:
                cnt = 1
                while stream_name+'!'+str(cnt) in streams:
                    cnt += 1
                streams[stream_name+'!'+str(cnt)] = streams[stream_name][:-1]
                streams[stream_name] = temp
        else:
            streams[stream_name] = temp
    #计算时间戳，确定客户端与服务器
    ret = dict()
    for k,v in streams.items():
        temp = k.split('!')
        if len(temp) > 1:
            k = temp[0]
            temp = '('+temp[1]+')'
        else:
            temp = ''
        s,us = v[0][1].split('.')
        s = int(s)
        us = int(us)
        for i in range(len(v)):
            si,usi = v[i][1].split('.')
            v[i][1] = int(si)-s + (int(usi)-us)*1e-6
        v = np.array(v)
        src, dst = k.split('-')
        #先根据端口号对客户端服务器进行矫正
        if (
            (int(src.split(':')[1]) < 1024 and int(dst.split(':')[1]) > 1024) or  #端口号判断
            (int(src.split(':')[1]) > 1024 and int(dst.split(':')[1]) > 1024 and  #端口号失效
            ip_type(src.split(':')[0]) == 'public' and ip_type(dst.split(':')[0]) == 'private') or #共私有IP判断
            (int(src.split(':')[1]) > 1024 and int(dst.split(':')[1]) > 1024 and  #端口号失效
            ip_type(src.split(':')[0]) == ip_type(dst.split(':')[0]) and  #共私有IP失效
            v[0][0] < 0)
        ): #第一个包当连接请求
            v[:,0] *= -1
            k = dst+'-'+src
        v[:,2:] *= v[:,:1]
        ret[temp+k] = v[:,1:]
    #返回文件中的TCP流
    return ret

def load_meta_from_csv(fname,start,end):
    #从csv文件中读取数据
    tcp_meta = []
    with open(fname,'r',encoding='utf-8') as f:
        for line in list(f)[start:end]:
            if line[-2] == '!':
                tcp_meta += [line[:-2]]
                
    #将数据以tcp流为单位进行组合
    streams = dict()
    for item in tcp_meta:
        #获取元数据中的各属性
        item_split = item.split(',')
        #获取流名称
        stream_name = item_split[tcp_stream_name]
        srcsock,dstsock = stream_name.split('-')
        stream_name = min(srcsock,dstsock)+'-'+max(srcsock,dstsock)
        temp = [[
            (float(srcsock < dstsock)-0.5) * 2, #方向先随便定义
            item_split[timestamp], #时间戳视为非数值，稍后单独处理
            float(item_split[ip_len]),
            float(item_split[payload_len]),
            float(item_split[UAPRSF_tcp_flags][2]=='1'),
        ]]
        if stream_name in streams:
            streams[stream_name] += temp
            t1 = float(streams[stream_name][-2][1])
            t2 = float(streams[stream_name][-1][1])
            if t2 < t1 or t2 - t1 > 300:
                cnt = 1
                while stream_name+'!'+str(cnt) in streams:
                    cnt += 1
                streams[stream_name+'!'+str(cnt)] = streams[stream_name][:-1]
                streams[stream_name] = temp
        else:
            streams[stream_name] = temp
    #计算时间戳，确定客户端与服务器
    ret = dict()
    for k,v in streams.items():
        temp = k.split('!')
        if len(temp) > 1:
            k = temp[0]
            temp = '('+temp[1]+')'
        else:
            temp = ''
        s,us = v[0][1].split('.')
        s = int(s)
        us = int(us)
        for i in range(len(v)):
            si,usi = v[i][1].split('.')
            v[i][1] = int(si)-s + (int(usi)-us)*1e-6
        v = np.array(v)
        src, dst = k.split('-')
        #先根据端口号对客户端服务器进行矫正
        if (
            (int(src.split(':')[1]) < 1024 and int(dst.split(':')[1]) > 1024) or  #端口号判断
            (int(src.split(':')[1]) > 1024 and int(dst.split(':')[1]) > 1024 and  #端口号失效
            ip_type(src.split(':')[0]) == 'public' and ip_type(dst.split(':')[0]) == 'private') or #共私断
            (int(src.split(':')[1]) > 1024 and int(dst.split(':')[1]) > 1024 and  #端口号失效
            ip_type(src.split(':')[0]) == ip_type(dst.split(':')[0]) and  #共私有IP失效
            v[0][0] < 0)
        ): #第一个包当连接请求
            v[:,0] *= -1
            k = dst+'-'+src
        v[:,2:] *= v[:,:1]
        ret[temp+k] = v[:,1:]
    #返回文件中的TCP流
    return ret
