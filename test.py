import torch
import time
import sys
torch.set_printoptions(precision=8)

def main():
    torch.set_num_threads(1)
    e2e_encoder = torch.jit.load(sys.argv[1])
    infeat = torch.ones(1, 86, 40).cuda()
    length = torch.IntTensor([86]).cuda()
    batch = 4
    infeat = torch.cat([infeat for i in range(batch)])
    length = torch.cat([length for i in range(batch)])
    print(infeat.shape, length)
    e2e_encoder(infeat , length)

if __name__ == '__main__':
    main()

