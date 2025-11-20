from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.Synthetic.GeneratorFactory import GeneratorFactory, GeneratorConfig, GeneratorType

# 1. Instantiate the main generator
synthetic_generator = SyntheticGenerator(random_state=42)

# 2. Configure and create the first Agrawal generator
print("Configuring generator for classification function 0...")
config1 = GeneratorConfig(classification_function=0, random_state=42)
agrawal_gen_1 = GeneratorFactory.create_generator(
    generator_type=GeneratorType.AGRAWAL,
    config=config1
)

# 3. Generate the first dataset
print("Generating dataset for function 0...")
synthetic_generator.generate(
    generator_instance=agrawal_gen_1,
    output_path='.',
    filename='agrawal_func0.csv',
    n_samples=10000,
    save_dataset=True,
    generate_report=False # Disable report for this example
)
print("Dataset 'agrawal_func0.csv' created.")

# 4. Configure and create the second Agrawal generator
print("\nConfiguring generator for classification function 2...")
config2 = GeneratorConfig(classification_function=2, random_state=42)
agrawal_gen_2 = GeneratorFactory.create_generator(
    generator_type=GeneratorType.AGRAWAL,
    config=config2
)

# 5. Generate the second dataset
print("Generating dataset for function 2...")
synthetic_generator.generate(
    generator_instance=agrawal_gen_2,
    output_path='.',
    filename='agrawal_func2.csv',
    n_samples=10000,
    save_dataset=True,
    generate_report=False # Disable report for this example
)
print("Dataset 'agrawal_func2.csv' created.")

print("\nBoth datasets have been generated successfully.")