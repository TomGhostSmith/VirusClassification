import json
import math
import threading
import subprocess
import fcntl
import time
from tqdm import tqdm

from utils import IOUtils
from config import config
from entity.sample import Sample
from entity.proteinSample import ProteinSample

class Worker:
    def __init__(self, cmd, updateResult):
        withShell = isinstance(cmd, str)
        self.proc = subprocess.Popen(cmd, shell=withShell, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.receiveThread = threading.Thread(target=self.receiveResult)
        self.receiveThread.start()
        self.submitted = 0
        self.finished = 0
        self.updateResult = updateResult
        self.sampleIDs = []

    def execute(self, message, sampleID=None):
        self.proc.stdin.write(message + "\n")
        if (sampleID is None):
            sampleID = self.submitted
        self.submitted += 1
        self.sampleIDs.append(sampleID)
        
    def receiveResult(self):
        for line in self.proc.stdout:
            line = line.strip("\n")
            if not line:
                continue
            self.updateResult(self.sampleIDs[self.finished], line)  # problem: one chunk per line?
            self.finished += 1

    def commit(self):
        self.proc.stdin.close()
        
    def join(self):
        self.proc.wait()
        self.receiveThread.join()

class WorkerPool:
    def __init__(self, scripts, desc=""):
        self.poolSize = len(scripts)
        self.workers = [Worker(script, self.update) for script in scripts]
        self.results = []
        self.bar = None
        self.desc = desc
        self.lock = threading.Lock()

    
    def update(self, sampleID, line):
        self.results[sampleID] = line  # this is thread safe even without GIL
        with self.lock:
            self.bar.update(1)

    def run(self, samples:list[Sample|ProteinSample], chunkSize=None): 
        if chunkSize is None:
            chunkSize = math.ceil(len(samples)/self.poolSize)
        self.results = [None] * len(samples)
        self.bar = tqdm(total=len(samples), desc=self.desc)

        if (isinstance(samples[0], Sample)):
            iterator = IOUtils.dumpSamples
        elif (isinstance(samples[0], ProteinSample)):
            iterator = IOUtils.dumpProteinSamples

        finished = 0
        while finished < len(samples):
            # check if any worker is available
            for worker in self.workers:
                if (worker.submitted == worker.finished):
                    # prepare the new chunk and let the worker run the chunk
                    end = min(finished + chunkSize, len(samples))
                    for idx, msg in zip(range(finished, end), iterator(samples[finished : end])):
                        worker.execute(msg, idx)
                    finished = end
                    break

            time.sleep(1)
        
        for worker in self.workers:
            worker.commit()

        for worker in self.workers:
            worker.join()

        self.bar.close()

        return self.results


CPULock = f"/tmp/CPULock"

def acquire_lock(lock_file_path):
    """Acquire a lock using a file-based locking mechanism."""
    while True:
        try:
            # Attempt to open the file in write mode
            lock_file = open(lock_file_path, 'w')
            # Try to acquire a non-blocking lock (using fcntl)
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except BlockingIOError:
            # If the lock file is already locked, wait for a while before retrying
            time.sleep(1)

def release_lock(lock_file):
    """Release the lock on the given lock file."""
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()