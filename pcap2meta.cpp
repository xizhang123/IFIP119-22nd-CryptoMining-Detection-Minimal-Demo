#include <pcap.h>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include <vector>
#include <string>

FILE* file;
void packet_handler(unsigned char* user_data, const struct pcap_pkthdr* pkthdr, const unsigned char* packet) {
    if (packet[0] == 0x45 || (packet[12] == 0x08 && packet[13] == 0x00 && packet[23] == 0x06)) {
        if (packet[0] == 0x45) { //only ip segment without frame head
            packet = packet - 14;
        }
        int total_len = (packet[16] << 8) + packet[17];
        int ip_head = (packet[14] & 0xF) * 4;
        int payload = total_len - ip_head - (packet[ip_head + 26] >> 4) * 4;
        fprintf(file, "%d.%d.%d.%d:%d-%d.%d.%d.%d:%d,",
            packet[26], packet[27], packet[28], packet[29], (packet[34] << 8) + packet[35],
            packet[30], packet[31], packet[32], packet[33], (packet[36] << 8) + packet[37]
        );
        fprintf(file, "%d.%06d,%d,%d,%d%d%d%d%d%d!\n",
            pkthdr->ts.tv_sec, pkthdr->ts.tv_usec, total_len, payload,
            (packet[ip_head + 27] >> 5) & 1,
            (packet[ip_head + 27] >> 4) & 1,
            (packet[ip_head + 27] >> 3) & 1,
            (packet[ip_head + 27] >> 2) & 1,
            (packet[ip_head + 27] >> 1) & 1,
            (packet[ip_head + 27] >> 0) & 1
        );
    }
}

std::string get_fname(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

std::string basename(const std::string& path) {
    size_t pos = path.find_last_of(".");
    return (pos == std::string::npos) ? path : path.substr(0,pos);
}

int main(int argc,char *argv[]) {
    char errbuf[PCAP_ERRBUF_SIZE]; // Error buffer
    for (int i = 1;i < argc;++i) {
        printf("out_fname:%s\n", (basename(get_fname(argv[i])) + ".csv").c_str());
        file = fopen((basename(get_fname(argv[i])) + ".csv").c_str(), "w");
        fprintf(file, "tcp_stream_name,timestamp,ip_len,payload_len,UAPRSF_tcp_flags\n");
        // Open the capture file for offline analysis
        printf("pcap_fname:%s\n", argv[i]);
        pcap_t* pcap = pcap_open_offline(argv[i], errbuf);
        if (pcap == NULL) {
            std::cerr << "Error opening file: " << errbuf << std::endl;
            fclose(file);
            return 1;
        }
        pcap_loop(pcap, -1, packet_handler, NULL);
        pcap_close(pcap);
        fclose(file);
    }
    return 0;
}
