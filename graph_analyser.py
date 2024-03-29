import networkx as nx
from networkx.algorithms import community
import numpy as np
import pandas as pd
import pickle
import re
from .helper import Helper
import seaborn as sns
import hbond_analyser
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GraphAnalyser():
	def __init__(self, pdb_code, pdb_file):
		self.pdb_code = pdb_code
		self.pdb_file = pdb_file
		self.helper = Helper()

	def create_graph(self, gtype='hba'):
		if gtype == 'hba':
			self.hba = self.helper.hbonds_from_pdb(self.pdb_file)
			self.graph = self.hba.filtered_graph
		elif gtype == 'wba':
			self.wba = self.helper.wwire_from_pdb(self.pdb_file)
			self.graph = self.wba.filtered_graph
		self.bw_data = hbond_analyser.load_bw_file(self.pdb_code)

	def construct_connected_component_details(self):
		self.model = self.helper.load_pdb_model(self.pdb_file)
		connected_components = list(nx.connected_components(self.graph))

		prot_chain = self.model[list(connected_components[0])[0][0]]

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

				if res_index <= len(self.bw_data):
					bw_number = hbond_analyser.get_bw_number(self.bw_data[res_index-1])
					tm = self.bw_data[res_index-1]['protein_segment']
					amino = self.bw_data[res_index-1]['amino_acid']
				else:
					bw_number = 1
					tm = ''
					amino = ''

				chain_details.append((res_name, coords, bw_number, amino, tm))
			all_chains.append(chain_details)

		self.connected_component_details = [c for c in sorted(all_chains, key=len, reverse=True)]


	def write_connected_components(self, folder):
		with open(folder+'/'+self.pdb_code+'_connected_components.pickle', 'wb') as fp:
			pickle.dump(self.connected_component_details, fp)


	def load_connected_components(self, folder):
		with open(folder+'/'+self.pdb_code+'_connected_components.pickle', 'rb') as fp:
			self.connected_component_details = pickle.load(fp)

	def closeness_centrality(self, folder=None):
		closeness_cent = dict(nx.closeness_centrality(self.graph))
		self.closeness_cent = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)
		if folder:
			cc = pd.DataFrame(data=self.closeness_cent, columns=['res_name', 'closeness_cent'])
			cc.to_csv(folder+'/'+self.pdb_code+'_closeness_cent.csv', index=False)

	def degree_centrality(self, folder=None):
		degree_cent = dict(nx.degree_centrality(self.graph))
		self.degree_cent = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
		if folder:
			dc = pd.DataFrame(data=self.degree_cent, columns=['res_name', 'degree_cent'])
			dc.to_csv(folder+'/'+self.pdb_code+'_degree_cent.csv', index=False)

	def betweenness_centrality(self, folder=None):
		betweenness_cent = dict(nx.betweenness_centrality(self.graph))
		self.betweenness_cent = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)
		if folder:
			bc = pd.DataFrame(data=self.betweenness_cent, columns=['res_name', 'betweenness_cent'])
			bc.to_csv(folder+'/'+self.pdb_code+'_betweenness_cent.csv', index=False)

	def write_centralities(self, folder=None):
		centralities = pd.DataFrame(data=self.betweenness_cent, columns=['res_name', 'betweenness_cent'])
		for i in self.degree_cent:
			centralities.loc[(centralities['res_name'] == i[0]), 'degree_cent'] = i[1]
		for i in self.closeness_cent:
			centralities.loc[(centralities['res_name'] == i[0]), 'closeness_cent'] = i[1]
		if folder:
			centralities.to_csv(folder+'/'+self.pdb_code+'_centralities.csv', index=False)
		self.centralities = centralities
		self.centralities.style.background_gradient(cmap='Greys')

	def _scale(self, col):
		normalized_col = (col-col.min())/(col.max()-col.min())
		return normalized_col

	def create_score_bw_centralities(self, folder=None):
		scored_centralities = self.centralities.copy()
		scored_centralities['betweenness_cent'] = self._scale(self.centralities['betweenness_cent'])
		scored_centralities['degree_cent'] = self._scale(self.centralities['degree_cent'])
		scored_centralities['closeness_cent'] = self._scale(self.centralities['closeness_cent'])
		scored_centralities['score'] = scored_centralities.sum(axis=1)
		bw_numbers = []
		tms = []
		amino_acid_code = []
		for index, row in scored_centralities.iterrows():
		    res_index = int(re.findall(r'\d+', row[0])[0])
		    if res_index <= len(self.bw_data):
		    	bw_number = hbond_analyser.get_bw_number(self.bw_data[res_index-1])
		    	tm = self.bw_data[res_index-1]['protein_segment']
		    	amino = self.bw_data[res_index-1]['amino_acid']
		    else:
		    	bw_number = 1
		    	tm = ''
		    	amino = ''
		    bw_numbers.append(bw_number)
		    tms.append(tm)
		    amino_acid_code.append(amino)

		scored_centralities['BW number'] = bw_numbers
		scored_centralities['TM'] = tms
		scored_centralities['amino_acid'] = amino_acid_code
		self.score_bw_centralities = scored_centralities.sort_values(by=['score'], ascending=False)
		self.score_bw_centralities.style.background_gradient(cmap='Greys')
		if folder:
			self.score_bw_centralities.to_csv(folder+'/'+self.pdb_code+'_score_bw_centralities.csv', index=False)

	def create_extended_table(self, folder):
		self.closeness_centrality()
		self.degree_centrality()
		self.betweenness_centrality()
		self.write_centralities()
		self.create_score_bw_centralities()

		extended_table = self.score_bw_centralities.copy()
		for i, chain in enumerate(self.connected_component_details):
			for item in chain:
				extended_table.loc[(extended_table['res_name'] == item[0]), 'res_id'] = int(re.findall(r'\d+', item[0])[0])
				extended_table.loc[(extended_table['res_name'] == item[0]), 'res_code'] = item[0][2:5]
				extended_table.loc[(extended_table['res_name'] == item[0]), 'x'] = item[1][0]
				extended_table.loc[(extended_table['res_name'] == item[0]), 'y'] = item[1][1]
				extended_table.loc[(extended_table['res_name'] == item[0]), 'z'] = item[1][2]
				extended_table.loc[(extended_table['res_name'] == item[0]), 'chain_id'] = i
				extended_table.loc[(extended_table['res_name'] == item[0]), 'chain_length'] = len(chain)

		extended_table.to_csv(folder+'/'+self.pdb_code+'_extended_table.csv', index=False)
		self.extended_table = extended_table

	def load_extended_table(self, folder):
		self.extended_table = pd.read_csv(folder+'/'+self.pdb_code+'_extended_table.csv')

	def filter_top_centralities(self, sort_by, top=30):
		df = self.extended_table.sort_values(by=[sort_by], ascending=False)
		df = df[(df['BW number'] > 1)]
		filtered_df = df[(df['BW number'] > 1)][['BW number', sort_by]].to_numpy()[:top]
		return filtered_df

	def add_avg_coords(self, coord_table, folder):
		et = self.extended_table
		for index, row in coord_table.iterrows():
			et.loc[(et['BW number'] == row['bw_number']), 'avg_x'] = row['avg_x']
			et.loc[(et['BW number'] == row['bw_number']), 'avg_y'] = row['avg_y']
			et.loc[(et['BW number'] == row['bw_number']), 'avg_z'] = row['avg_z']

		et.to_csv(folder+'/'+self.pdb_code+'_extended_table.csv', index=False)
		self.extended_table = et

	def add_communities(self, folder):
		et = self.extended_table
		communities_generator = community.girvan_newman(self.graph)
		top_level_communities = next(communities_generator)
		next_level_communities = next(communities_generator)
		self.communities = sorted(map(sorted, next_level_communities))
		for i, comm in enumerate(self.communities):
			for res_name in comm:
				et.loc[(et['res_name'] == res_name), 'community_id'] = i

		et.to_csv(folder+'/'+self.pdb_code+'_extended_table.csv', index=False)
		self.extended_table = et


	def plot_projected_centrality_score(self, folder, color_by='score'):
		#here use self.extended_table or just make code pretty everywhere else
		extended_table = self.extended_table.sort_values(by=[color_by], ascending=False)
		XY = extended_table[['x', 'y']].to_numpy()
		z = extended_table['z'].to_numpy()
		pca = PCA(n_components=1)
		xy = pca.fit_transform(XY)
		color = extended_table[color_by].to_numpy()
		labels = extended_table['BW number'].to_numpy()
		fig, ax = plt.subplots(figsize=(13,13))
		ax.scatter(xy, z, c=color, cmap='YlGn')
		for i, text in enumerate(labels[:35]):
		    ax.annotate(text, (xy[i], z[i]))

		ax.set_xlabel('projected xy plane')
		ax.set_ylabel('Z-axis')
		ax.set_title(self.pdb_code + ' projected '+color_by)
		fig.savefig(folder+'/'+self.pdb_code+'_projected_centrality_'+color_by+'.png')

	def tm_members_of_chains(self):
		data = np.zeros(shape=(7, len(self.connected_component_details)))
		for i, chain in enumerate(self.connected_component_details):
			for item in chain:
				numb = re.findall(r'\d+\.', str(item[2]))
				if len(numb) > 0 and int(numb[0][:-1]) <= 7 and int(numb[0][0]) > 0: 
					#create heatmap datamatrix
					tm = numb[0][:-1]
					data[int(tm)-1, i] +=1
		return data

	def plot_TM_members_of_chains(self, folder_name=None):
		fig, ax = plt.subplots(figsize=(8, 10))

		data = self.tm_members_of_chains()
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

		data = self.interpolate_Z_axis()

		mask = np.zeros_like(data)
		mask[data == 0] = True
		ax = sns.heatmap(data, square=True, cbar=False, annot=True, cmap='Blues', mask=mask)

		ax.set_xticklabels([len(c) for c in self.connected_component_details])
		# ax.set_yticklabels(mesh, rotation=360)
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

	def get_transformed_graph_positions(self, score, coords):
		nodes = self.graph.nodes
		node_positions = {}
		edges = list(self.graph.edges)
		average_water_per_wire = self.wba.compute_average_water_per_wire()

		for index, row in self.extended_table.iterrows():
		    res = str(row['res_name'])
		    if res in nodes:
		        node_positions[res] = {'bw_number': row['BW number'],
		                              'id': row['res_id'],
		                              'score': row[score],
		                              'amino_acid':row['amino_acid'],
		                              'tm': row['TM'],
		                              'community_id': row['community_id']}

		        if coords == 'avg' and int(node_positions[res]['bw_number']) != 0:
		        	node_positions[res]['coords'] = [row['avg_x'], row['avg_y'], row['avg_z']]
		        else:
		        	node_positions[res]['coords'] = [row['x'], row['y'], row['z']]
		
		XY = [i['coords'][0:2] for i in node_positions.values()]
		pca = PCA(n_components=1)
		xy = pca.fit_transform(XY)

		for i, item in enumerate(node_positions.values()):
		    item['pca'] = [xy[i][0], item['coords'][2]]

		edge_lines = []
		waters = []
		for edge in edges:
			edge_lines.append((node_positions[edge[0]]['pca'], node_positions[edge[1]]['pca']))
			key = str(edge[0])+':'+str(edge[1])
			if key in average_water_per_wire: 
				waters.append(average_water_per_wire[key])
			else:
				key = str(edge[1])+':'+str(edge[0])
				waters.append(average_water_per_wire[key])

		return node_positions, edge_lines, waters


	def plot_scored_bw_water_wire(self, folder_name, score='degree_cent', coords='unique', label=True):
		node_positions, edge_lines, waters = self.get_transformed_graph_positions(score, coords)

		plt.figure(figsize=(10,16))
		color = [item['score'] for item in node_positions.values()]
		x = [item['pca'][0] for item in node_positions.values()]
		y = [item['pca'][1] for item in node_positions.values()]

		# plt.scatter(x, y, c='#39465e', s=300)
		
		plt.scatter(x, y, c=color, cmap='Blues', s=300)
		for i, item in enumerate(node_positions.values()):
			if int(str(item['bw_number'])[0]) == 0:
				plt.annotate(item['tm'], (item['pca'][0], item['pca'][1]+0.25), weight='bold')
			else:
				plt.annotate('{:.2f}'.format(float(item['bw_number'])), (item['pca'][0], item['pca'][1]+0.25), weight='bold')
			plt.annotate(str(item['amino_acid'])+str(int(item['id'])), (item['pca'][0]+0.2, item['pca'][1]-0.25))
		    

		for i, edge in enumerate(edge_lines):
		    x=[edge[0][0], edge[1][0]]
		    y=[edge[0][1], edge[1][1]]
		    
		    plt.plot(x, y, c='gray')
		    # plt.plot(x, y, c='#424242')
		    if label:
		        plt.annotate(int(waters[i]), (x[0] + (x[1]-x[0])/2, y[0] + (y[1]-y[0])/2), color='indianred')


		plt.title(self.pdb_code+' water wire')
		# plt.axis('off')
		plt.xlabel('Projected xy plane')
		plt.ylabel('Z-axis')
		plt.savefig(folder_name+'/'+self.pdb_code+'_'+score+'_bw_water_wire_'+coords+'_coords_bare.png')


	def plot_water_wire_communities(self, folder_name, score='degree_cent', coords='unique', label=True):
		self.add_communities(folder_name)
		node_positions, edge_lines, waters = self.get_transformed_graph_positions(score, coords)


		plt.figure(figsize=(10,16))

		viridis = plt.cm.get_cmap('tab20', 20)
		clrs = viridis(np.linspace(0, 1, 20))
		colors = np.concatenate((clrs[::2], clrs[1::2]))

		grays = ['dimgrey', 'grey', 'darkgray', 'silver', 'lightgray', 'lightgray']
		for i, edge in enumerate(edge_lines):
		    x=[edge[0][0], edge[1][0]]
		    y=[edge[0][1], edge[1][1]]
		    
		    plt.plot(x, y, linewidth=1.3, color=grays[int(waters[i])])

		for item in node_positions.values():
			plt.scatter(item['pca'][0], item['pca'][1], color=colors[int(item['community_id'])],  s=50+item['score']*300, label=int(item['community_id']))


		for i, item in enumerate(node_positions.values()):
			if int(str(item['bw_number'])[0]) == 0:
				plt.annotate(item['tm'], (item['pca'][0], item['pca'][1]+0.25), weight='bold')
			else:
				plt.annotate('{:.2f}'.format(float(item['bw_number'])), (item['pca'][0], item['pca'][1]+0.25), weight='bold')
			plt.annotate(str(item['amino_acid'])+str(int(item['id'])), (item['pca'][0]+0.2, item['pca'][1]-0.25))
		   

		plt.title(self.pdb_code+' water wire')
		# plt.legend()
		# plt.axis('off')
		plt.xlabel('Projected xy plane')
		plt.ylabel('Z-axis')
		plt.savefig(folder_name+'/'+self.pdb_code+'_'+score+'_bw_water_wire_'+coords+'_communities.png')