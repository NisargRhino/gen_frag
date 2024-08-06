import os
from rdkit import Chem
from rdkit.Chem import Recap, AllChem
from tqdm import tqdm
import pandas as pd
from dockstring import load_target
import argparse
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(filename='process_log.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def fragment_molecule_recaps(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            logging.warning(f"Failed to parse SMILES: {smiles}")
            return []

        recap_tree = Recap.RecapDecompose(molecule)
        fragments = []

        if recap_tree:
            leaves = recap_tree.GetLeaves()
            if leaves:
                for smile, node in leaves.items():
                    cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                    fragments.append(cleaned_smile)
            else:
                logging.warning("No leaves found in the Recap tree.")
        else:
            logging.warning("Failed to obtain Recap decomposition.")
        
        return fragments

    except Exception as e:
        logging.error(f"Error during fragmentation: {e}")
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
        logging.error(f"Error during molecule cleanup: {e}")
        return None

def dock_fragment(frag, target, docking_dir, mol2_path, center_coords, box_sizes):
    try:
        cleaned_mol = cleanup_molecule_rdkit(frag)
        if cleaned_mol is None:
            return None, float('inf')

        cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
        score, __ = target.dock(cleaned_smiles)

        return cleaned_smiles, score
    except Exception as e:
        logging.error(f"Error docking fragment {frag}: {e}")
        return None, float('inf')

def dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    os.makedirs(docking_dir, exist_ok=True)

    convert_command = f"obabel -imol2 {mol2_path} -opdbqt -O {os.path.join(docking_dir, target_name + '_target.pdbqt')} -xr"
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

    # Parallel processing of fragment docking
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(dock_fragment, frag, target, docking_dir, mol2_path, center_coords, box_sizes) for frag in fragments]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Docking fragments"):
            try:
                cleaned_smiles, score = future.result()
                if score < best_score:
                    best_score = score
                    best_fragment = cleaned_smiles
            except Exception as e:
                logging.error(f"Error in future result: {e}")

    return best_fragment, best_score

def process_single_drug(smiles, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    try:
        fragments = fragment_molecule_recaps(smiles)
        if not fragments:
            return None

        best_fragment, best_score = dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes)
        if best_fragment is not None:
            return {'SMILES': smiles, 'BestFragment': best_fragment, 'BestScore': best_score}

    except Exception as e:
        logging.error(f"Error processing drug with SMILES {smiles}: {e}")
    
    return None

def main(input_csv, mol2_path, docking_dir, target_name, center_coords, box_sizes, output_path):
    try:
        input_data = pd.read_csv(input_csv)
        input_data.columns = input_data.columns.str.strip()

        results = []
        batch_size = 10

        for index, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Processing drugs"):
            smiles = row.get('SMILES', None)
            if smiles is not None:
                result = process_single_drug(smiles, target_name, docking_dir, mol2_path, center_coords, box_sizes)
                if result is not None:
                    results.append(result)
            
            # Save results in batches
            if (index + 1) % batch_size == 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_path, index=False)
                print(f"Intermediate results saved after {index + 1} drugs.")

        # Save any remaining results after the loop
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"Final results saved to {output_path}.")
    
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug fragmentation, cleanup, and docking script')

    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with SMILES strings')
    parser.add_argument('--mol2_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--docking_dir', type=str, default='dockdir', help='Docking directory name/path')
    parser.add_argument('--target_name', type=str, default='target', help='Target name')
    parser.add_argument('--center_coords', type=float, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, nargs=3, help='Box sizes for docking (X Y Z)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the results CSV')

    args = parser.parse_args()

    main(args.input_csv, args.mol2_path, args.docking_dir, args.target_name, args.center_coords, args.box_sizes, args.output_path)
