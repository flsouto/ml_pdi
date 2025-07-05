import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|predict]")
        return

    command = sys.argv[1]

    if command == "train":
        subprocess.run(["python", "train.py"])
    elif command == "predict":
        subprocess.run(["python", "predict.py"])
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train|predict]")

if __name__ == "__main__":
    main()