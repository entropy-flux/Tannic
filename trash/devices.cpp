#include <stdio.h>

struct host_t {
    enum traits { 
        PAGGED = 1 << 0,
        PINNED = 1 << 1,
        MAPPED = 1 << 2
    };
    traits flag;
};

struct device_t {
    int id;
    enum flags {
        SYNC = 1 << 0,
        ASYNC = 1 << 1, 
    };
    flags flag;
};

void print_host_flags(struct host_t h) {
    printf("Host flags: ");
    if (h.flag & host_t::PAGGED) printf("PAGGED ");
    if (h.flag & host_t::PINNED) printf("PINNED ");
    if (h.flag & host_t::MAPPED) printf("MAPPED ");
    printf("\n");
}

void print_device_flags(struct device_t d) {
    printf("Device %d flags: ", d.id);
    if (d.flag & device_t::SYNC)  printf("SYNC ");
    if (d.flag & device_t::ASYNC) printf("ASYNC ");
    printf("\n");
}

int main() {
    host_t h = { static_cast<host_t::traits>(host_t::PAGGED | host_t::MAPPED) };
    print_host_flags(h);

    h.flag = static_cast<host_t::traits>(h.flag | host_t::PINNED);  
    print_host_flags(h);  

    if (h.flag & host_t::MAPPED) {
        printf("Host is mapped!\n"); 
    }

    struct device_t d = { .id = 42, .flag = device_t::ASYNC };
    print_device_flags(d);  

    d.flag = static_cast<device_t::flags>(d.flag ^ device_t::SYNC);     
    print_device_flags(d);

    return 0;
}