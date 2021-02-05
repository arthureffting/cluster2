import sys
import time

print("[Initial alignment]")

if len(sys.argv) >= 3 and sys.argv[2] == "init":
    for i in range(10):
        time.sleep(1)
        print(i)
