

def main():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NuScenes", help="Dataset type (e.g., NuScenes)")
    parser.add_argument("--version", type=str, default='v1.0-mini', help="Version of dataset")
    args = parser.parse_args()

    # Check for CUDA
    if (not torch.cuda.is_available()):
        print(colored("CUDA is unavailable! Closing program"), "red")
        quit()

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB total")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # Load dataset folder
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "datasets", args.dataset)
    print(f"Dataset root: {dataset_root}")    



if __name__ == "__main__":
    main()