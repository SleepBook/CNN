#include "CNNcpu.h"

int main()
{
	int layer_num;
	FILE *net_config;
	FILE *input_data;

	float *****filters;
	int **kernel_dim;
	float **bias;
	int *feature_num;
	int **inter_res_dim;//attention feature_num and inter_res_dim include the input data
	float ****inter_res;
	int *layer_type;//internal code for layers: CONV 0 FULL 1 POOL 2

	int i, j, k, m, n;

	//readin the net
	net_config = fopen("test.cnet", "r");

	fscanf(net_config, "%d", &layer_num);
	layer_type = (int*)malloc(sizeof(int)*layer_num);
	feature_num = (int*)malloc(sizeof(int)*(layer_num+1));
	kernel_dim = (int**)malloc(sizeof(int*)*layer_num);
	inter_res_dim = (int**)malloc(sizeof(int*)*(layer_num + 1));
	bias = (float**)malloc(sizeof(float*)*layer_num);	
	inter_res = (float****)malloc(sizeof(float***)*layer_num);
	filters = (float*****)malloc(sizeof(float****)*layer_num);	
	for (i = 0; i < layer_num; i++)
	{
		inter_res_dim[i] = (int*)malloc(sizeof(int) * 2);
		kernel_dim[i] = (int*)malloc(sizeof(int) * 2);//assume each layer has uniformed kernel size
	}
	inter_res_dim[layer_num] = (int*)malloc(sizeof(int)*2);
	feature_num[0] = 1;
	inter_res_dim[0][0] = 32;
	inter_res_dim[0][1] = 32;//0 represent width, 1 represent hight
	
	for (i = 0; i < layer_num; i++)
	{
		fscanf(net_config, "%d", &layer_type[i]);
		if (layer_type[i] == 0)
		{
			printf("read in a CONV layer\n");
			int flt_size, front_feature_size, flt_w, flt_h, stride;
			fscanf(net_config, "%d", &flt_size);
			fscanf(net_config, "%d", &front_feature_size);
			fscanf(net_config, "%d", &flt_w);
			fscanf(net_config, "%d", &flt_h);
			fscanf(net_config, "%d", &stride);

			filters[i] = (float****)malloc(sizeof(float***)*flt_size);
			kernel_dim[i][0] = flt_w;
			kernel_dim[i][1] = flt_h;
			bias[i] = (float*)malloc(sizeof(float)*flt_size);
			feature_num[i+1] = flt_size;
			inter_res_dim[i + 1][0] = inter_res_dim[i][0] - flt_w + 1;
			inter_res_dim[i + 1][1] = inter_res_dim[i][1] - flt_h + 1;
			//allocating space to store inter result
			inter_res[i] = (float***)malloc(sizeof(float**)*flt_size);
			for (j = 0; j < flt_size; j++)
			{
				inter_res[i][j] = (float**)malloc(sizeof(float*)*inter_res_dim[i + 1][1]);
				for (m = 0; m < inter_res_dim[i + 1][1]; m++)
				{
					inter_res[i][j][m] = (float*)malloc(sizeof(float)*inter_res_dim[i + 1][0]);
				}
			}
			
			//readin the kernel group
			for (j = 0; j < flt_size; j++)
			{
				filters[i][j] = (float***)malloc(sizeof(float**)*front_feature_size);
				//filters[i][j] = (float***)malloc(sizeof(float**)*flt_w*flt_h);
				for (k = 0; k < front_feature_size; k++)
				{
					filters[i][j][k] = NULL;
				}
				int flt_num = 0;
				fscanf(net_config, "%d", &flt_num);
				//printf("the fltnum read in is %d\n",flt_num);
				for (k = 0; k < flt_num; k++)
				{
					int flt_squence;
					fscanf(net_config, "%d", &flt_squence);
					filters[i][j][flt_squence] = (float**)malloc(sizeof(float*) * flt_h);
					for (m = 0; m < flt_h; m++)
					{
						filters[i][j][flt_squence][m] = (float*)malloc(sizeof(float) * flt_w);
						for (n = 0; n < flt_w; n++)
						{
							fscanf(net_config, "%f", &filters[i][j][flt_squence][m][n]);
						}
					}
				}
				//scan the bias item
				fscanf(net_config, "%f", &bias[i][j]);
			}
		}
		else if (layer_type[i] == 1)
		{
			printf("read in a FULL layer\n");
		}
		else if (layer_type[i] == 2)
		{
			printf("read in a pooling layer\n");

		}
		else
		{
			printf("unrecognized layer percepted, exiting now\n");
			return -1;
		}
	}
	
	printf("readin the net done\n");
	//show the kernels readin
	for (i = 0; i < layer_num; i++)
	{
		printf("\n\nshow layer %d :\n", i);
		printf("this layer has %d kernel groups:\n", feature_num[i + 1]);
		for (j = 0; j < feature_num[i + 1]; j++)
		{
			printf("\tprint %d kernel group info:\n", j);
			for (k = 0; k < feature_num[i]; k++)
			{
				if (filters[i][j][k] != NULL)
				{
					printf("\t\tthis kernel has a link to previous feature map %d\n", k);
					for (m = 0; m < kernel_dim[i][1]; m++)
					{
						printf("\t\t\t");
						for (n = 0; n < kernel_dim[i][0]; n++)
						{
							printf("%f, ", filters[i][j][k][m][n]);
						}
						printf("\n");
					}
					printf("\t\tthe bias for this kernel is: %f\n", bias[i][j]);
				}
			}
		}
	}
	

	//recycle the mem

	//free mem store the internal result
	free(inter_res_dim[0]);
	for (i = 0; i < layer_num; i++)
	{
		for (j = 0; j < feature_num[i + 1]; j++)
		{
			for (k = 0; k < inter_res_dim[i][1];k++)
			{
				free(inter_res[i][j][k]);
			}
			free(inter_res[i][j]);
		}
		free(inter_res[i]);
		free(inter_res_dim[i+1]);
	}
	free(inter_res_dim);
	free(inter_res);

	//free the kernels
	for (i = 0; i < layer_num; i++)
	{
		for (j = 0; j < feature_num[i + 1]; j++)
		{
			for (k = 0; k < feature_num[i]; k++)
			{
				if (filters[i][j][k] != NULL)
				{
					for (m = 0; m < kernel_dim[i][1]; m++)
					{
						free(filters[i][j][k][m]);
					}
					free(filters[i][j][k]);
				}
			}
			free(filters[i][j]);
		}
		free(filters[i]);
		free(bias[i]);
		free(kernel_dim[i]);
	}
	free(filters);
	free(bias);
	free(kernel_dim);

	//free others
	free(layer_type);
	free(feature_num);
	fclose(net_config);
	
	return 0;
}