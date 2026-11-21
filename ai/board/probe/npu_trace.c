// LD_PRELOAD tracer for the RKNPU DRM ioctls. Logs MEM_CREATE/MEM_MAP, and for every SUBMIT the
// per-core task ranges plus the register command words of each task. Read-only observer.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>

#define NMEM 8192
struct memrec { uint32_t handle; uint64_t obj, dma, size, off; void *ptr; int used; };
static struct memrec mem[NMEM];
static int nmem;
static FILE *lf;
static int nsubmit;
static int max_submit = 100000;

struct rknpu_mem_create { uint32_t handle, flags; uint64_t size, obj_addr, dma_addr, sram_size; int32_t iommu_domain_id; uint32_t core_mask; };
struct rknpu_mem_map { uint32_t handle, reserved; uint64_t offset; };
struct rknpu_task { uint32_t flags, op_idx, enable_mask, int_mask, int_clear, int_status, regcfg_amount, regcfg_offset; uint64_t regcmd_addr; } __attribute__((packed));
struct sub { uint32_t task_start, task_number; };
struct rknpu_submit { uint32_t flags, timeout, task_start, task_number, task_counter; int32_t priority; uint64_t task_obj_addr; uint32_t iommu_domain_id, reserved; uint64_t task_base_addr; int64_t hw_elapse_time; uint32_t core_mask; int32_t fence_fd; struct sub subcore_task[5]; };

static void init(void) {
    if (lf) return;
    const char *p = getenv("NPU_TRACE_OUT");
    lf = fopen(p ? p : "/tmp/npu_trace.log", "w");
    if (getenv("NPU_TRACE_MAXSUBMIT")) max_submit = atoi(getenv("NPU_TRACE_MAXSUBMIT"));
    setvbuf(lf, NULL, _IOFBF, 1 << 20);
}
static struct memrec *by_handle(uint32_t h) { for (int i = nmem - 1; i >= 0; i--) if (mem[i].used && mem[i].handle == h) return &mem[i]; return NULL; }
static struct memrec *by_obj(uint64_t o) { for (int i = nmem - 1; i >= 0; i--) if (mem[i].used && mem[i].obj == o) return &mem[i]; return NULL; }
static struct memrec *by_dma(uint64_t a) { for (int i = nmem - 1; i >= 0; i--) if (mem[i].used && a >= mem[i].dma && a < mem[i].dma + mem[i].size) return &mem[i]; return NULL; }
static struct memrec *by_off(uint64_t o) { for (int i = nmem - 1; i >= 0; i--) if (mem[i].used && mem[i].off == o && o) return &mem[i]; return NULL; }

int ioctl(int fd, unsigned long req, ...) {
    static int (*real)(int, unsigned long, ...);
    if (!real) real = dlsym(RTLD_NEXT, "ioctl");
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    unsigned type = (req >> 8) & 0xff, nr = req & 0xff;
    if (type == 'd' && nr >= 0x41 && nr <= 0x45) {
        init();
        int idx = nr - 0x40;  // 1 submit, 2 create, 3 map, 4 destroy, 5 sync
        if (idx == 1 && nsubmit < max_submit) {
            struct rknpu_submit *s = arg;
            fprintf(lf, "SUBMIT n=%d core_mask=0x%x task_start=%u task_number=%u base=0x%llx obj=0x%llx sub=", nsubmit, s->core_mask, s->task_start, s->task_number, (unsigned long long)s->task_base_addr, (unsigned long long)s->task_obj_addr);
            for (int i = 0; i < 5; i++) fprintf(lf, "[%u,%u]", s->subcore_task[i].task_start, s->subcore_task[i].task_number);
            fprintf(lf, "\n");
            struct memrec *tm = by_obj(s->task_obj_addr);
            if (tm && tm->ptr) {
                struct rknpu_task *t = tm->ptr;
                // dump the union of all subcore ranges
                for (int c = 0; c < 5; c++) {
                    unsigned st = s->subcore_task[c].task_start, nn = s->subcore_task[c].task_number;
                    for (unsigned k = 0; k < nn; k++) {
                        struct rknpu_task *q = &t[st + k];
                        fprintf(lf, "TASK sub=%d idx=%u op=%u flags=0x%x en=0x%x intm=0x%x amount=%u offset=%u regcmd=0x%llx\n", c, st + k, q->op_idx, q->flags, q->enable_mask, q->int_mask, q->regcfg_amount, q->regcfg_offset, (unsigned long long)q->regcmd_addr);
                        struct memrec *rm = by_dma(q->regcmd_addr);
                        if (rm && rm->ptr) {
                            uint64_t *w = (uint64_t *)((char *)rm->ptr + (q->regcmd_addr - rm->dma));
                            unsigned n = q->regcfg_amount + 4;
                            if (n > 400) n = 400;
                            fprintf(lf, "CMD");
                            for (unsigned j = 0; j < n; j++) fprintf(lf, " %016llx", (unsigned long long)w[j]);
                            fprintf(lf, "\n");
                        } else fprintf(lf, "CMD unmapped\n");
                    }
                }
            } else fprintf(lf, "TASKOBJ unmapped\n");
            nsubmit++;
            fflush(lf);
        }
        int r = real(fd, req, arg);
        if (idx == 2 && r == 0 && nmem < NMEM) {
            struct rknpu_mem_create *c = arg;
            mem[nmem++] = (struct memrec){c->handle, c->obj_addr, c->dma_addr, c->size, 0, NULL, 1};
        } else if (idx == 3 && r == 0) {
            struct rknpu_mem_map *m = arg; struct memrec *x = by_handle(m->handle);
            if (x) x->off = m->offset;
        }
        return r;
    }
    return real(fd, req, arg);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = dlsym(RTLD_NEXT, "mmap");
    void *p = real(addr, len, prot, flags, fd, off);
    if (p != MAP_FAILED && fd >= 0 && off) { struct memrec *x = by_off((uint64_t)off); if (x) x->ptr = p; }
    return p;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) { return mmap(addr, len, prot, flags, fd, off); }

__attribute__((destructor)) static void stats(void) {
    if (!lf) return;
    int withoff = 0, withptr = 0;
    for (int i = 0; i < nmem; i++) { if (mem[i].off) withoff++; if (mem[i].ptr) withptr++; }
    fprintf(lf, "STATS nmem=%d withoff=%d withptr=%d submits=%d\n", nmem, withoff, withptr, nsubmit);
    fflush(lf);
}
