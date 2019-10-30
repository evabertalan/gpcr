import os
import pickle
from shutil import copyfile
import numpy as np
from mdhbond import HbondAnalysis
from mdhbond import WireAnalysis
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Helper:
    def __init__(self):
        pass

    def create_directory(self, directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        return directory


    def get_files(self, folder, endswith):
            return [file for file in os.listdir(folder) if file.endswith(endswith)]
        

    def get_pdb_files(self, folder):
        return [file for file in os.listdir(folder) if file.endswith('.pdb')]


    def no_pdb_files(self, folder):
        pdb_files = get_pdb_files(folder)
        if len(pdb_files) == 0:
            print("No pdb file in the folder")
            return True
        else:    
            print("There are pdb files in the folder")
            return False

        
    def load_pdb_model(self, pdb_file):
    	parser = PDBParser(QUIET=True)
    	structure = parser.get_structure('model', pdb_file)
    	return structure[0]


    def hbonds_from_pdb(self, pdb_file):
        hba = HbondAnalysis('protein',
                            pdb_file, 
                            residuewise=True, 
                            check_angle=False, 
                            additional_donors=['N'], 
                            additional_acceptors=['O'], 
                            add_donors_without_hydrogen=True)
        hba.set_hbonds_in_selection(exclude_backbone_backbone=True)
        if len(self.water_in_pdb(pdb_file)) > 0:
            hba.set_hbonds_in_selection_and_water_around(5.0)
        hba.filter_occupancy(0.1, use_filtered=False)
        return hba

    def wwire_from_pdb(self, pdb_file):
        wwa = WireAnalysis('protein',
                pdb_file, 
                residuewise=True, 
                check_angle=False,
                add_donors_without_hydrogen=True)
        wwa.set_water_wires(max_water=5)
        wwa.compute_average_water_per_wire()
        return wwa


    def dump_hba_to_pickle(self, hba, target_file):
        with open(target_file, 'wb') as fp:
            pickle.dump(hba, fp, pickle.HIGHEST_PROTOCOL)

        
    def load_hba_from_pickle(self, source_file):
        with open(source_file, 'rb') as fp:
            hba = pickle.load(fp)
        return hba

        
    def water_in_pdb(self, pdb_file):
        # pdb_file should be: directory+'/'+code+'.pdb'
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure('model', pdb_file)
        residues = list(struct[0].get_residues())
        waters = [res for res in residues if res.get_id()[0] == 'W']
        return waters


    def water_number_in_pdb(self, pdb_file):
        waters = water_in_pdb(pdb_file)
        return len(waters)


    def water_in_chain(self, pdb_file, chain_id):
        parser = PDBParser(QUIET = True)
        struct = parser.get_structure('model', pdb_file)
        chain = list(struct[0][chain_id].get_residues())
        waters = [chain[i] for i in range(len(chain)) if chain[i].get_id()[0] == 'W']
        return waters


    def water_coordinates(self, pdb_file, chain_id=None):
        if chain_id:
            waters = water_in_chain(pdb_file, chain_id)
        else:
            waters = water_in_pdb(pdb_file)
        
        water_coord = [water['O'].get_coord() for water in waters]
        return np.array(water_coord)


    def normalized_internal_water_coordinates(self, file, folder):
        io = PDBIO()
        parser = PDBParser(QUIET = True)
        
        code = file[0:4]
        struct = parser.get_structure(code, folder+file)

        plane = list(struct[0][' '].get_residues())
        up_plane = plane[1]['O'].get_coord()[2]
        down_plane = plane[0]['N'].get_coord()[2]
        
        if len(list(list(struct[0])[0].get_residues())) > 100:
            chain = list(struct[0])[0]
        else:
            chain = list(struct[0])[1]
        
        
        chain_id = chain.get_id()
        waters = water_in_chain(folder+file, chain_id)
        internal_waters = [water for water in waters if water['O'].get_coord()[2] < up_plane and water['O'].get_coord()[2] > down_plane]

        internal_water_coords = [water['O'].get_coord() for water in internal_waters]
        norm_coords = internal_water_coords/up_plane
        return norm_coords


    class TMSelect(Select):
        def __init__(self, up_plane, down_plane, chain):
            self.up_plane = up_plane
            self.down_plane = down_plane
            self.chain = chain
            
        def accept_residue(self, res):
            if (res.has_id('CA') and res['CA'].get_coord()[2] < self.up_plane 
                and res['CA'].get_coord()[2] > self.down_plane 
                and res.get_parent().id == self.chain.id):
                return 1
            
            elif (list(res)[0].get_coord()[2] < self.up_plane 
                  and list(res)[0].get_coord()[2] > self.down_plane 
                  and res.get_parent().id == self.chain.id):
                return 1
            
            else:
                return 0
            
    def has_correct_dum_in_pdb(self, file_path):
        io = PDBIO()
        parser = PDBParser(QUIET = True)
        code = file_path[-8:-4]
        struct = parser.get_structure(code, file_path)
        try:
            list(struct[0][' '].get_residues())
        except KeyError:
            print('Error in file:', file_path)
            
    def get_longest_chain(self, chains):
        max_chain_length = 0
        
        for i in range(len(chains)):
            current_length = len(list(chains[i].get_residues()))
            if  current_length > max_chain_length and current_length < 700:
                max_chain_length = current_length
                longest_chain = chains[i]
        return longest_chain

            
    def get_TM_chain(self, chains, up_plane, down_plane):
        #select that chain which has the most residues in the TM
        tm_chain = []
        
        for chain in chains:
            residues = list(chain.get_residues())
            for res in residues:
                if (res.has_id('CA') and res['CA'].get_coord()[2] < up_plane 
                and res['CA'].get_coord()[2] > down_plane):
                    tm_chain.append(res.get_parent().id)

        unique_chain, count = np.unique(np.array(tm_chain), return_counts=True)
        return unique_chain[count == max(count)][0]


    def cut_7_helix(self, source_pdb, target_folder, offset=0):
        io = PDBIO()
        parser = PDBParser(QUIET = True)
        
        code = source_pdb[-8:-4]
        struct = parser.get_structure(code, source_pdb)

        plane = list(struct[0][' '].get_residues())
        
        try:
            up_plane = plane[1]['O'].get_coord()[2] + offset
        except KeyError:
            up_plane = plane[0]['O'].get_coord()[2] + offset
        
        try:
            down_plane = plane[0]['N'].get_coord()[2] - offset
        except KeyError:
            down_plane = plane[1]['N'].get_coord()[2] - offset
        
        
        chain_id = get_TM_chain(list(struct[0]), up_plane, down_plane)
        chain = struct[0][chain_id]

        io.set_structure(struct)
        io.save(target_folder+code+'_tm.pdb', TMSelect(up_plane, down_plane, chain))
        

    def concatenate_arrays(self, arrays):
        conc = []
        for arr in arrays:  
            if arr.ndim > 1:
                for row in arr:
                    conc.append(row)
            elif arr.size != 0:
                conc.append(arr)
        return np.array(conc)  


    def copy_pdb_file(self, pdb_code, source_folder, dest_folder):
        source_file = [file for file in os.listdir(source_folder) if file.startswith(pdb_code)]
        source = source_folder+source_file[0]
        dest = dest_folder+source_file[0]
        copyfile(source, dest)
        

    def plot_water_Z_axis(self, waters, round_to=0):
        unique_z, counts_z = np.unique(np.round(waters[:,2], round_to), return_counts=True)
        
        fig = plt.figure(figsize=(10,8))
        plt.plot(counts_z, unique_z, '.', color='maroon')
        plt.plot(counts_z, unique_z, linewidth=0.7, color='gray')
        plt.title('Number of waters in crystal structures along the Z axis', fontsize=20)
        plt.xlabel('Number of waters', fontsize=20)
        plt.ylabel('Z-axis coordinate', fontsize=20)
        return fig


    def histogram_water_Z_axis(self, waters):
        fig = plt.figure(figsize=(10,8))
        plt.hist(waters[:,2], color='lightsteelblue', bins=40, orientation='horizontal')
        plt.title('Number of waters in crystal structures along the Z axis')
        plt.xlabel('Number of waters')
        plt.ylabel('Z-axis coordinate')
        return fig



    def sodium_in_pdb(self, pdb_file):
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure('model', pdb_file)
        residues = list(struct[0].get_residues())
        sodium = [res for res in residues if res.get_id()[0] == 'SOD' or res.get_id()[0] == 'H_ NA']

        return sodium
        
        
    def sodium_coordinate(self, pdb_file):
        sods = sodium_in_pdb(pdb_file)
        sod_coord = [sod['NA'].get_coord() for sod in sods]
        return np.array(sod_coord)


    def plot_water_sodium_Z_axis(self, waters, sodiums):
        unique_w, counts_w = np.unique(np.round(waters[:,2]), return_counts=True)
        unique_s, counts_s = np.unique(np.round(sodiums[:,2]), return_counts=True)
        
        fig = plt.figure(figsize=(10,8))
        plt.plot(counts_w, unique_w, '.', color='maroon')
        plt.plot(counts_w, unique_w, linewidth=0.8, color='gray')
        
        plt.plot(counts_s, unique_s, '.', color='purple')
        plt.plot(counts_s, unique_s, linewidth=0.7, color='gray')
        
        plt.title('Number of waters and sodium in crystal structures along the Z axis',  fontsize=20)
        plt.xlabel('Number',  fontsize=20)
        plt.ylabel('Z-axis coordinate',  fontsize=20)
        return fig
        
        
    def histogram_water_sodium_Z_axis(self, waters, sodiums):
        fig = plt.figure(figsize=(10,8))
        plt.hist(waters[:,2], color='lightsteelblue', bins=40, orientation='horizontal')
        plt.hist(sodiums[:,2], color='purple', bins=20, orientation='horizontal')
        plt.title('Number of waters in crystal structures along the Z axis')
        plt.xlabel('Number of waters')
        plt.ylabel('Z-axis coordinate')
        return fig
        
    def plot_water_3d(self, waters):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection="3d")
        ax.scatter3D(waters[:,0], waters[:,1], waters[:,2], s=0.3, c='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig

    def plot_clusters(self, waters, labels, cc):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection="3d")
        ax.scatter3D(waters[:, 0], waters[:, 1], waters[:, 2],
                   c=labels.astype(np.float))
        ax.scatter3D(cc[:, 0], cc[:, 1], cc[:, 2], color='k', marker='*', s=100)
        ax.set_title('Water clusters with k='+str(len(cc)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig

        
    def plot_cluster_centers(self, cc):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection="3d")
        ax.scatter3D(cc[:, 0], cc[:, 1], cc[:, 2], color='r', s=100, marker='o')
        ax.set_title('Water cluster centers with k='+str(len(cc)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig