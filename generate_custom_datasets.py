import pandas as pd
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from river.datasets.synth import Agrawal

def generate_dataset1():
    """
    Generates the first dataset with 6000 instances in 6 blocks of 1000.
    All blocks are generated with seed 422 and classification_function=0.
    """
    print("Generating Dataset 1...")
    generator = SyntheticGenerator(random_state=422)
    agrawal_gen = Agrawal(seed=422, classification_function=0)

    df = generator.generate(
        generator_instance=agrawal_gen,
        output_path=".",
        filename="dataset1.csv",
        n_samples=6000,
        target_col="class",
        save_dataset=False,
        generate_report=False
    )

    # Add block column
    df['block'] = (df.index // 1000) + 1
    
    df.to_csv("dataset1.csv", index=False)
    print("Dataset 1 generated and saved to dataset1.csv")

def generate_dataset2():
    """
    Generates the second dataset with 6000 instances in 6 blocks of 1000.
    Blocks 4 and 6 have a different classification function.
    """
    print("Generating Dataset 2...")
    generator = SyntheticGenerator(random_state=123)
    
    # Generator for blocks 1, 2, 3, 5
    agrawal_gen1 = Agrawal(seed=123, classification_function=0)
    
    # Generator for blocks 4, 6
    agrawal_gen2 = Agrawal(seed=123, classification_function=1)
    
    all_blocks = []
    
    for i in range(1, 7):
        print(f"  Generating block {i}...")
        gen_instance = agrawal_gen2 if i in [4, 6] else agrawal_gen1
        
        df_block = generator.generate(
            generator_instance=gen_instance,
            output_path=".",
            filename=f"temp_block_{i}.csv", # Temporary file
            n_samples=1000,
            target_col="class",
            save_dataset=False,
            generate_report=False
        )
        df_block['block'] = i
        all_blocks.append(df_block)
        
    df_final = pd.concat(all_blocks, ignore_index=True)
    df_final.to_csv("dataset2.csv", index=False)
    print("Dataset 2 generated and saved to dataset2.csv")


if __name__ == "__main__":
    generate_dataset1()
    generate_dataset2()
