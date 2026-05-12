from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

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
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        assert src_array.size % nprocs == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % nprocs == 0, "dest_array size must be divisible by the number of processes"

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



