from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.training import train
if __name__=='__main__':
    result=train(ROOT/'artifacts')
    print('Training completed:', result)
