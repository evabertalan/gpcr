import networkx as nx
import numpy as np
import pickle
import re
import helper
import seaborn as sns
import hbond_analyser
import matplotlib.pyplot as plt


class GraphAnalyser():
	def __init__(self, pdb_code, pdb_file):
		self.pdb_code = pdb_code
		self.pdb_file = pdb_file

	def create_graph(self):
		self.hba = helper.hbonds_from_pdb(self.pdb_file)
		self.graph = self.hba.filtered_graph

	def construct_connected_component_details(self):
		self.model = helper.load_pdb_model(self.pdb_file)
		connected_components = list(nx.connected_components(self.graph))

		prot_chain = self.model[list(connected_components[0])[0][0]]
		bw_data = hbond_analyser.load_bw_file(self.pdb_code)

		all_chains = []
		for connected_chain in connected_components:
			connected_chain = list(connected_chain)
			chain_details = []
			for res_name in connected_chain:
				res_index = int(re.findall(r'\d+', res_name)[0])

				if re.search('HOH', res_name): coords = prot_chain[('W', res_index, ' ')]['O'].get_coord()
				else:
					res = prot_chain[res_index]
					coords = res['CA'].get_coord()

				if res_index <= len(bw_data): bw_number = hbond_analyser.get_bw_number(bw_data[res_index-1])
				else: bw_number = 1

				# if re.search('HOH', res_name): bw_number=1
				# else:  bw_number = 0

				
				chain_details.append((res_name, coords, bw_number))
			all_chains.append(chain_details)

		self.connected_component_details = [c for c in sorted(all_chains, key=len, reverse=True)]


	def write_connected_components(self, folder):
		with open(folder+'/'+self.pdb_code+'_connected_components.pickle', 'wb') as fp:
			pickle.dump(self.connected_component_details, fp)


	def load_connected_components(self, folder):
		with open(folder+'/'+self.pdb_code+'_connected_components.pickle', 'rb') as fp:
			self.connected_component_details = pickle.load(fp)


	def tm_members_of_chains(self):
		data = np.zeros(shape=(7, len(self.connected_component_details)))
		for i, chain in enumerate(self.connected_component_details):
			for item in chain:
				numb = re.findall(r'\d+\.', str(item[2]))
				if len(numb) > 0 and int(numb[0][:-1]) <= 7 and int(numb[0][0]) > 0: 
					#create datamatrix for heatmap
					tm = numb[0][:-1]
					data[int(tm)-1, i] +=1
		return data

	def plot_TM_members_of_chains(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(8, 10))

		data = tm_members_of_chains()
		mask = np.zeros_like(data)
		mask[data == 0] = True
		ax = sns.heatmap(data, linewidth=0.5, cmap="Greens",square=True, annot=True, mask=mask, cbar=False)

		ax.set_xticklabels([len(c) for c in self.connected_component_details])
		ax.set_yticklabels([str(i) + ' TM' for i in np.arange(data.shape[0])+1], rotation=360)
		ax.set_xlabel('Lenght of chain')
		ax.set_ylabel('Transmembrane helix')
		ax.set_title(self.pdb_code + ' occurance of TMs in chains')
		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_TM_members_of_chains_heatmap.png')


	def interpolate_Z_axis(self):
		all_z_coords = []
		for chain in self.connected_component_details:
			all_z_coords = all_z_coords + [item[1][2] for item in chain]
		mesh = np.round(np.linspace(max(all_z_coords), min(all_z_coords), 10, endpoint=True), 2)

		data = np.zeros(shape=(len(mesh), len(self.connected_component_details)))
		for i, chain in enumerate(self.connected_component_details):
			z_coords = [item[1][2] for item in chain if not re.search('HOH', item[0])]

			near_point = [mesh[(np.abs(mesh-coord)).argmin()]  for coord in z_coords]
			for point in near_point:
				data[np.where(mesh == point)[0][0], i] += 1
		return data

	def plot_interploated_Z_axis(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(8, 10))

		data = interpolate_Z_axis()

		mask = np.zeros_like(data)
		mask[data == 0] = True
		ax = sns.heatmap(data, square=True, cbar=False, annot=True, cmap='Blues', mask=mask)

		ax.set_xticklabels([len(c) for c in self.connected_component_details])
		ax.set_yticklabels(mesh, rotation=360)
		ax.set_xlabel('Lenght of chain')
		ax.set_ylabel('Interpolated Z axis coordinates')
		ax.set_title(self.pdb_code+' number of nodes in the interpolated distance')
		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_interpolated_z_axis_heatmap.png')


	def plot_3_longest_chains_along_z(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(8, 75))

		for i, chain in enumerate(self.connected_component_details[0:3]):
		    for item in chain:
		        z_coords = item[1][2]
		        color = '#b3453d' if item[2] == 1 else '#39465e'
		        ax.scatter(i, z_coords, color=color, s=20)
		        ax.annotate(item[2], (i, z_coords))

		ax.set_xticks(np.arange(len(self.connected_component_details[0:3])))
		ax.set_xticklabels([len(c) for c in self.connected_component_details[0:3]])
		ax.set_xlabel('# of nodes in the Hbond chain')
		ax.set_ylabel('Z-axis coordinate')
		ax.set_title(self.pdb_code+' continius Hbond chains wit BW names along the Z-axis')

		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_3_longest_chains_along_z.png')



	def plot_chians_along_Z(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(8, 10))

		for i, chain in enumerate(self.connected_component_details):
		    for item in chain:
		        z_coords = item[1][2]
		        color = '#b3453d' if item[2] == 1 else '#39465e'
		        ax.scatter(i, z_coords, color=color, s=20)
		        
		ax.set_xticks(np.arange(len(self.connected_component_details)))
		ax.set_xticklabels([len(c) for c in self.connected_component_details])
		ax.set_xlabel('# of nodes in the Hbond chain')
		ax.set_ylabel('Z-axis coordinate')
		ax.set_title(self.pdb_code+' continius Hbond chains along the Z-axis')

		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_chains_along_z.png')


	def plot_nodes_in_TM_along_Z(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(5, 10))
		
		for chain in self.connected_component_details:
		    chain = sorted(chain, key=lambda item: int(str(item[2])[0]))
		    for item in chain:
		        z_coords = item[1][2]
		        numb = re.findall(r'\d+\.', str(item[2]))
		        if len(numb) > 0 and int(numb[0][:-1]) <= 8: 
		            tm = numb[0][:-1]
		            color = '#b3453d' if item[2] == 1 else '#39465e'
		            ax.scatter(tm, z_coords, color=color, s=20)
		ax.set_ylabel('Z-axis coordinate')
		ax.set_xlabel('TM helix')
		ax.set_title(self.pdb_code+' Hbond nodes in TMs along the Z-axis')

		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_nodes_in_TM_along_Z.png')

	def plot_bw_in_TM_along_Z(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(10, 16))

		for chain in self.connected_component_details:
		    chain = sorted(chain, key=lambda item: int(str(item[2])[0]))

		    for item in chain:
		        z_coords = item[1][2]
		        numb = re.findall(r'\d+\.', str(item[2]))
		        if len(numb) > 0 and int(numb[0][:-1]) <= 8: 
		            tm = numb[0][:-1]
		            color = '#b3453d' if item[2] == 1 else '#39465e'
		            ax.scatter(tm, z_coords, color=color, s=17)
		            ax.annotate(item[2], (tm, z_coords))
		ax.set_ylabel('Z-axis coordinate')
		ax.set_xlabel('TM helix')
		ax.set_title(self.pdb_code+' Hbond nodes in TMs along the Z-axis')

		
		if folder_name:
			fig.savefig(folder_name+'/'+self.pdb_code+'_bw_number_TM_along_Z.png')

# def degree_centrality(): 

