"""MPI wrapper for PA3 Part 1.

This wrapper exposes both the buffered NumPy-style collectives that you used
in PA2 (`Allreduce`, `Allgather`, `Reduce_scatter`, `Alltoall`) and the
pickle-based Python-object collectives (`bcast`, `allgather`, `alltoall`, ...).

In Part 1 you will mostly use the pickle-based variants because token routing
in a Mixture-of-Experts produces variable-sized payloads per rank. The buffered
variants are still available if you want to use them in optimized code paths.

If you want to drop in your own implementations of all-reduce / all-to-all from
PA2 (Section 2.1), copy `myAllreduce` and `myAlltoall` from your PA2
`mpi_wrapper/comm.py` into the marked locations below. Doing so is optional but
recommended for the EP implementation; see the bonus rubric in the README.
"""

from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    # ---------- basic info ----------
    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    # ---------- pickle-based (Python object) collectives ----------
    def bcast(self, data, root=0):
        return self.comm.bcast(data, root=root)

    def allgather(self, data):
        return self.comm.allgather(data)

    def alltoall(self, send_data):
        return self.comm.alltoall(send_data)

    def allreduce(self, data, op=MPI.SUM):
        return self.comm.allreduce(data, op=op)

    # ---------- buffered (NumPy) collectives ----------
    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_bytes = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_bytes * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()
        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
        self.comm.Alltoall(src_array, dest_array)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    # ---------- optional: paste your PA2 implementations here ----------
    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        Your implementation should respect the passed reduction operator.
        The required operators for this assignment are MPI.MIN, MPI.SUM,
        and MPI.MAX.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        assert src_array.size == dest_array.size

        nprocs = self.comm.Get_size()  # number of processes
        rank = self.comm.Get_rank()  # rank of the current process
        root = 0

        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size

        if rank == root:
            # Root process receives data from all other processes and applies reduction.
            reduced = np.array(src_array, copy=True)
            temp = np.empty_like(src_array)

            for src in range(nprocs):
                if src == root:
                    continue

                self.comm.Recv(temp, source=src, tag=0)

                if op == MPI.SUM:
                    reduced += temp
                elif op == MPI.MIN:
                    np.minimum(reduced, temp, out=reduced)
                elif op == MPI.MAX:
                    np.maximum(reduced, temp, out=reduced)
                else:
                    raise ValueError("Unsupported reduction operator")

            np.copyto(dest_array, reduced)

            for dest in range(nprocs):
                if dest == root:
                    continue
                self.comm.Send(dest_array, dest=dest, tag=0)
        else:
            # Non-root processes send their data to the root and receive the reduced result.
            self.comm.Send(src_array, dest=root, tag=0)
            self.comm.Recv(dest_array, source=root, tag=0)
            self.total_bytes_transferred += src_array_byte + dest_array_byte

    def myAlltoall(self, src_array, dest_array):
        """Manual all-to-all using pairwise Sendrecv exchanges."""
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        send_count = src_array.size // nprocs
        recv_count = dest_array.size // nprocs
        assert send_count == recv_count

        src_flat = src_array.reshape(-1)
        dest_flat = dest_array.reshape(-1)

        for other in range(nprocs):
            send_start = other * send_count
            send_end = send_start + send_count
            recv_start = other * recv_count
            recv_end = recv_start + recv_count

            if other == rank:
                np.copyto(dest_flat[recv_start:recv_end], src_flat[send_start:send_end])
            else:
                self.comm.Sendrecv(
                    sendbuf=src_flat[send_start:send_end],
                    dest=other,
                    sendtag=rank,
                    recvbuf=dest_flat[recv_start:recv_end],
                    source=other,
                    recvtag=other,
                )

        send_seg_bytes = src_array.itemsize * send_count
        recv_seg_bytes = dest_array.itemsize * recv_count
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)


# Default global communicator (mirrors the pa2 convention).
mpi = Communicator(MPI.COMM_WORLD)
