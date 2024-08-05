import os
from rdkit import Chem
from rdkit.Chem import Recap, AllChem
from tqdm import tqdm
import pandas as pd
from dockstring import load_target
import argparse
from concurrent.futures import ProcessPoolExecutor


def fragment_molecule_recaps(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"Warning: Failed to parse SMILES: {smiles}")
            return []

        recap_tree = Recap.RecapDecompose(molecule)
        fragments = []

        if recap_tree:
            leaves = recap_tree.GetLeaves()
            if leaves:
                for smile, node in leaves.items():
                    # Properly handle wildcard atoms
                    cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                    fragments.append(cleaned_smile)
                    print(f"Fragment SMILES: {cleaned_smile}")
            else:
                print("No leaves found in the Recap tree.")
        else:
            print("Failed to obtain Recap decomposition.")
        
        return fragments

    except Exception as e:
        print(f"Error during fragmentation: {e}")
        return []

def cleanup_molecule_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        return mol

    except Exception as e:
        print(f"Error during molecule cleanup: {e}")
        return None

def dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    os.makedirs(docking_dir, exist_ok=True)

    convert_command = f"obabel -imol2 {mol2_path} -opdbqt -O {os.path.join(docking_dir, target_name + '_target.pdbqt')} -xr"
    print(f"Running command: {convert_command}")
    os.system(convert_command)

    conf_path = os.path.join(docking_dir, target_name + '_conf.txt')
    with open(conf_path, 'w') as f:
        f.write(f"""center_x = {center_coords[0]}
center_y = {center_coords[1]}
center_z = {center_coords[2]}

size_x = {box_sizes[0]}
size_y = {box_sizes[1]}
size_z = {box_sizes[2]}""")

    target = load_target(target_name, targets_dir=docking_dir)

    best_score = float('inf')  # Initialize to positive infinity
    best_fragment = None

    for frag in fragments:
        try:
            cleaned_mol = cleanup_molecule_rdkit(frag)
            if cleaned_mol is None:
                continue  # Skip docking if cleaning failed

            cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
            score, __ = target.dock(cleaned_smiles)

            if score < best_score:  # Update condition for lower scores being better
                best_score = score
                best_fragment = cleaned_smiles
        except Exception as e:
            print(f"Error docking fragment {frag}: {e}")

    return best_fragment, best_score

def process_row(row, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    smiles = row.get('SMILES', None)
    if smiles is None:
        return None

    fragments = fragment_molecule_recaps(smiles)
    if not fragments:
        return None

    best_fragment, best_score = dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes)
    if best_fragment is not None:
        return {'Name': row['name'], 'SMILES': smiles, 'BestFragment': best_fragment, 'BestScore': best_score}
    
    return None

def process_batch(batch, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_row, row, target_name, docking_dir, mol2_path, center_coords, box_sizes) for _, row in batch.iterrows()]
        
        for future in tqdm(futures, total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)

    return results

def main(input_csv, mol2_path, docking_dir, target_name, center_coords, box_sizes, output_path, batch_size):
    input_data = pd.read_csv(input_csv)
    input_data.columns = input_data.columns.str.strip()
    results = []

    num_batches = len(input_data) // batch_size + int(len(input_data) % batch_size != 0)

    for i in range(num_batches):
        batch = input_data[i*batch_size:(i+1)*batch_size]
        batch_results = process_batch(batch, target_name, docking_dir, mol2_path, center_coords, box_sizes)
        results.extend(batch_results)

    if not results:
        print("No successful docking results.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug fragmentation, cleanup, and docking script')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with SMILES strings')
    parser.add_argument('--mol2_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--docking_dir', type=str, default='dockdir', help='Docking directory name/path')
    parser.add_argument('--target_name', type=str, default='target', help='Target name')
    parser.add_argument('--center_coords', type=float, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, nargs=3, help='Box sizes for docking (X Y Z)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the results CSV')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')

    args = parser.parse_args()
    main(args.input_csv, args.mol2_path, args.docking_dir, args.target_name, args.center_coords, args.box_sizes, args.output_path, args.batch_size)
