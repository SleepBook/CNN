#include "CNNcpu.h"

float non_linear(int type, float input_num)
{
	if (type == 0)
	{
		return 1.0 / (1.0 + exp(0.0 - input_num));
	}
	
}

int run_cpu()
{
	int layer_num;
	FILE *net_config;
	FILE *input_data;

	int *layer_type;//internal code for layers: CONV 0 FULL 1 POOL 2
	int *feature_num;//include the input data, n+1

	float *****filters;
	int **kernel_dim;
	int *stride;
	int *method;
	float **bias;
	
	int **inter_res_dim;//include the input data, n+1
	float ****inter_res;//same, n+1
	
	int i, j, k, m, n;

	//readin the net
	net_config = fopen("test.cnet", "r");
	fscanf(net_config, "%d", &layer_num);
	layer_type = (int*)malloc(sizeof(int)*layer_num);
	feature_num = (int*)malloc(sizeof(int)*(layer_num+1));
	kernel_dim = (int**)malloc(sizeof(int*)*layer_num);
	stride = (int*)malloc(sizeof(int)*layer_num);
	method = (int*)malloc(sizeof(int)*layer_num);
	inter_res_dim = (int**)malloc(sizeof(int*)*(layer_num + 1));
	bias = (float**)malloc(sizeof(float*)*layer_num);	
	inter_res = (float****)malloc(sizeof(float***)*(layer_num+1));
	filters = (float*****)malloc(sizeof(float****)*layer_num);	
	for (i = 0; i < layer_num; i++)
	{
		//the convention for the x_dim varible is, [0] specify the width, [1]specify the height
		inter_res_dim[i] = (int*)malloc(sizeof(int) * 2);
		inter_res_dim[i][0] = 0;
		inter_res_dim[i][1] = 0;
		kernel_dim[i] = (int*)malloc(sizeof(int) * 2);//assume at present each layer has uniformed kernel size
	}
	inter_res_dim[layer_num] = (int*)malloc(sizeof(int)*2);
	

	//for lenet
	inter_res_dim[0][0] = INPUT_WIDTH;
	inter_res_dim[0][1] = INPUT_HEIGHT;
	feature_num[0] = INPUT_CHANNEL;
	
	inter_res[0] = (float***)malloc(sizeof(float**)*feature_num[0]);
	inter_res[0][0] = (float**)malloc(sizeof(float*)*inter_res_dim[0][1]);
	for(i=0;i<inter_res_dim[0][1];i++)
	{
		inter_res[0][0][i] = (float*)malloc(sizeof(float)*inter_res_dim[0][0]);
	}
	
	for (i = 0; i < layer_num; i++)
	{
		fscanf(net_config, "%d", &layer_type[i]);
		if (layer_type[i] == 0)
		{
			if (DEBUG_MODE)
			{
				printf("read in a CONV layer\n");
			}
			int flt_size, front_feature_size, flt_w, flt_h, lstride;
			fscanf(net_config, "%d", &flt_size);
			fscanf(net_config, "%d", &front_feature_size);
			fscanf(net_config, "%d", &flt_w);
			fscanf(net_config, "%d", &flt_h);
			fscanf(net_config, "%d", &lstride);

			filters[i] = (float****)malloc(sizeof(float***)*flt_size);
			kernel_dim[i][0] = flt_w;
			kernel_dim[i][1] = flt_h;
			stride[i] = lstride;
			method[i] = -1;
			bias[i] = (float*)malloc(sizeof(float)*flt_size);
			feature_num[i+1] = flt_size;
			inter_res_dim[i + 1][0] = (inter_res_dim[i][0] - flt_w + 1)/lstride;
			inter_res_dim[i + 1][1] = (inter_res_dim[i][1] - flt_h + 1)/lstride;//the user has the responsibility to make sure it can zhengchu

			//allocating space to store inter result
			inter_res[i+1] = (float***)malloc(sizeof(float**)*flt_size);
			for (j = 0; j < flt_size; j++)
			{
				inter_res[i+1][j] = (float**)malloc(sizeof(float*)*inter_res_dim[i + 1][1]);
				for (m = 0; m < inter_res_dim[i + 1][1]; m++)
				{
					inter_res[i+1][j][m] = (float*)malloc(sizeof(float)*inter_res_dim[i + 1][0]);
					for(n=0;n<inter_res_dim[i+1][0];n++)
					{
						inter_res[i+1][j][m][n] = 0.0;
					}
				}
			}
			
			//readin the kernel group
			for (j = 0; j < flt_size; j++)
			{
				filters[i][j] = (float***)malloc(sizeof(float**)*front_feature_size);
				for (k = 0; k < front_feature_size; k++)
				{
					filters[i][j][k] = NULL;
				}
				int flt_num = 0;
				fscanf(net_config, "%d", &flt_num);
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
			if (DEBUG_MODE)
			{
				printf("read in a FULL layer\n");
			}
			feature_num[i + 1] = 1;
			stride[i] = -1;
			method[i] = -1;
			int front_feature_num = feature_num[i];
			int input_num, output_num;
			fscanf(net_config, "%d", &input_num);
			fscanf(net_config, "%d", &output_num);
			inter_res_dim[i+1][0] = output_num;
			inter_res_dim[i+1][1] = 1;		

			kernel_dim[i][0] = output_num;
			kernel_dim[i][1] = input_num;
			filters[i] = (float****)malloc(sizeof(float***) * 1);
			filters[i][0] = (float***)malloc(sizeof(float**) * front_feature_num);
			for (j = 0; j < front_feature_num; j++)
			{
				if (j == 0)
				{
					filters[i][0][j] = (float**)malloc(sizeof(float*)*input_num);
				}
				else
				{
					filters[i][0][j] = NULL;
				}
			}
			filters[i][0][0] = (float**)malloc(sizeof(float*)*input_num);
			bias[i]=(float*)malloc(sizeof(float)*output_num);
			for (j = 0; j < input_num; j++)
			{
				filters[i][0][0][j] = (float*)malloc(sizeof(float)*output_num);
			}
			for(j=0;j<input_num;j++)
			{
				for(k=0;k<output_num;k++)
				{
					fscanf(net_config, "%f", &filters[i][0][0][j][k]);
				}
			}
			for(k=0;k<output_num;k++)
			{
				fscanf(net_config, "%f", &bias[i][k]);
			}
		
			inter_res[i+1] = (float***)malloc(sizeof(float**) * 1);
			inter_res[i+1][0] = (float**)malloc(sizeof(float*) * 1);
			inter_res[i+1][0][0] = (float*)malloc(sizeof(float)*output_num);
			for (j = 0; j < output_num; j++)
			{
				inter_res[i + 1][0][0][j] = 0.0;
			}

		}
		else if (layer_type[i] == 2)
		{
			if (DEBUG_MODE)
			{
				printf("read in a pooling layer\n");
			}
			int lmethod = 0;//default methid is MAX
			stride[i] = -1;
			int front_feature_size, width, height;
			fscanf(net_config, "%d", &lmethod);
			fscanf(net_config, "%d", &front_feature_size);
			fscanf(net_config, "%d", &width);
			fscanf(net_config, "%d", &height);

			inter_res_dim[i + 1][0] = inter_res_dim[i][0] / width;
			inter_res_dim[i + 1][1] = inter_res_dim[i][1] / height;
			method[i] = lmethod;

			inter_res[i+1] = (float***)malloc(sizeof(float**) * front_feature_size);
			for (j = 0; j < front_feature_size; j++)
			{
				inter_res[i+1][j] = (float**)malloc(sizeof(float*)*inter_res_dim[i + 1][1]);
				for (k = 0; k < inter_res_dim[i + 1][1]; k++)
				{
					inter_res[i+1][j][k] = (float*)malloc(sizeof(float)*inter_res_dim[i + 1][0]);
					for (m = 0; m < inter_res_dim[i + 1][0];m++)
					{
						inter_res[i + 1][j][k][m] = 0.0;
					}
				}
			}

			filters[i] = NULL;
			bias[i] = NULL;
			feature_num[i+1] = front_feature_size;

			kernel_dim[i][0] = width;
			kernel_dim[i][1] = height; 
		}
		else
		{
			printf("unrecognized layer percepted, exiting now\n");
			if (DEBUG_MODE)
			{
				getchar();
			}
			return -1;
		}
	}
	fclose(net_config);

	if (DEBUG_MODE)
	{
		printf("readin the net done\n");
		//show the kernels readin

		for (i = 0; i < layer_num; i++)
		{
			if (i == 4)
			{
				printf("\n\nshow layer %d :\n", i);
				printf("this layer has %d kernel groups:\n", feature_num[i + 1]);
				if (filters[i] != NULL){
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
								if (i != 4)
								{
									printf("\t\tthe bias for this kernel is: %f\n", bias[i][j]);
								}
								else
								{
									printf("\t\tthe bias is: ");
									for (m = 0; m < 120; m++)
									{
										printf("%f ", bias[i][m]);
									}
									printf("\n");
								}
							}
						}
					}
				}
				else
				{
					printf("this is a pool layer\n");
					printf("the pooling window is %d * %d \n", kernel_dim[i][0], kernel_dim[i][1]);
				}
			}
		}
	}
	
	//start execution

	//readin the raw image first	
		input_data = fopen("test.cdat", "r");
		int input_channel, input_width, input_height;
		fscanf(input_data, "%d", &input_channel);
		fscanf(input_data, "%d", &input_width);
		fscanf(input_data, "%d", &input_height);
		if (feature_num[0] != input_channel || inter_res_dim[0][0] != input_width || inter_res_dim[0][1] != input_height)
		{
			printf("input data mismatch with the net, exiting now\n");
			if (DEBUG_MODE)
			{
				getchar();
			}
			return -1;
		}
		for (i = 0; i < input_channel; i++)
		{
			for (j = 0; j < input_height; j++)
			{
				for (k = 0; k < input_width; k++)
				{
					fscanf(input_data, "%f", &inter_res[0][i][j][k]);
				}
			}
		}
		if (DEBUG_MODE)
		{
			printf("read in the input image done\n");
		}
		fclose(input_data);
	
	
	//main execution iteration
	for(i=0;i<layer_num;i++)
	{
		if(layer_type[i]==0)
		{
			if (DEBUG_MODE){
				printf("execution convlution\n");
			}
			int pre_feature, next_feature;
			int ker_w, ker_h,strid;
			int res_w, res_h;
			res_w = inter_res_dim[i + 1][0];
			res_h = inter_res_dim[i + 1][1];
			pre_feature = feature_num[i];
			next_feature = feature_num[i+1];
			ker_w = kernel_dim[i][0];
			ker_h = kernel_dim[i][1];
			strid = stride[i];
			for(j=0;j<next_feature;j++)
			{				
				for(k=0;k<pre_feature;k++)
				{
					if(filters[i][j][k]!=NULL)
					{						
						for(m=0;m<res_h;m++)
						{
							for(n=0;n<res_w;n++)
							{
								float tmp_res= 0.0;
								int x, y;
								for(x=0;x<ker_h;x++)
								{
									for(y=0;y<ker_w;y++)
									{
										tmp_res += filters[i][j][k][x][y]*inter_res[i][k][m*strid+x][n*strid+y];
									}
								}
								inter_res[i+1][j][m][n] += tmp_res;
							}
						}

					}
				}
				for(m=0;m<res_h;m++)
				{
					for(n=0;n<res_w;n++)
					{
						inter_res[i+1][j][m][n] = non_linear(0,inter_res[i+1][j][m][n] + bias[i][j]);
					}
				}
			}
			if(DEBUG_MODE)
			{
				printf("show internal result on layer %d\n",i+1);
				for(j=0;j<next_feature;j++)
				{
					int res_w, res_h;
					res_w = inter_res_dim[i+1][0];
					res_h = inter_res_dim[i+1][1];
					printf("feature map %d ===============================================\n",j);
					for(m=0;m<res_h;m++)
					{
						for(n=0;n<res_w;n++)
						{
							printf("%f ",inter_res[i+1][j][m][n]);
						}
						printf("\n");
					}
				}
			}
		}
		else if(layer_type[i]==2)
		{
			if (DEBUG_MODE)
			{
				printf("execution pooling\n");
			}
			int ker_w, ker_h;
			ker_w = kernel_dim[i][0];
			ker_h = kernel_dim[i][1];
			int lmethod = method[i];
			float result;
			int x, y;
			for(j=0;j<feature_num[i];j++)
			{
				for(m=0;m<inter_res_dim[i+1][1];m++)
				{
					for(n=0;n<inter_res_dim[i+1][0];n++)
					{
						result = 0.0;
						for(x=0;x<ker_h;x++)
						{
							for(y=0;y<ker_w;y++)
							{
								if (lmethod==0)
								{
									if(inter_res[i][j][m*ker_h+x][n*ker_w+y]>result)
									{
										result = inter_res[i][j][m*ker_h+x][n*ker_w+y];
									}
								}
								else if (lmethod == 1)
								{
									result += inter_res[i][j][m*ker_h + x][n*ker_w + y];
								}
							}
						}
						if (lmethod == 1)
						{
							result = result / (ker_h*ker_w);
						}
						inter_res[i + 1][j][m][n] = result;
					}
				}
			}
			if (DEBUG_MODE)
			{
				printf("show internal result on layer %d\n", i + 1);
				for (j = 0; j < feature_num[i + 1]; j++)
				{
					int res_w, res_h;
					res_w = inter_res_dim[i + 1][0];
					res_h = inter_res_dim[i + 1][1];
					printf("feature map %d ===============================================\n", j);
					for (m = 0; m<res_h; m++)
					{
						for (n = 0; n<res_w; n++)
						{
							printf("%f ", inter_res[i + 1][j][m][n]);
						}
						printf("\n");
					}
				}
			}
		}
		else
		{
			if (DEBUG_MODE)
			{
				printf("excution FULL conection\n");
			}
			//first flatten the front data
			float* local_input = NULL;
			float local_res = 0.0;
			if(inter_res_dim[i][1]!=1 || feature_num[i]!=1)
			{
				int pre_feature, pre_res_w, pre_res_h,input_dim;
				pre_feature = feature_num[i];
				pre_res_w = inter_res_dim[i][0];
				pre_res_h = inter_res_dim[i][1];
				
				input_dim = pre_res_h*pre_res_w*pre_feature;
				if(input_dim != kernel_dim[i][1])
				{
					printf("FULL layer mismatch with previous dimension, exiting now\n");
					printf("%d %d %d\n", pre_feature, pre_res_h, pre_res_w);
					printf("%d", input_dim);
					if (DEBUG_MODE)
					{
						getchar();
					}
					return -1;
				}
				local_input = (float*)malloc(sizeof(float)*input_dim);
				for(j=0;j<pre_feature;j++)
				{			
					for (m = 0; m < pre_res_h; m++)
					{
						for (n = 0; n < pre_res_w; n++)
						{
							local_input[j*pre_res_h*pre_res_w + m*pre_res_w + n] = inter_res[i][j][n][m];
							//attention here, the reason for [n][m] not [m][n] is the netdata i retrieved from matlab
							//flat the data by coloum first, not row first, that's 1 2 is flatted into 1 3 2 4 
							//                                                     3 4
						}
					}
				}
				for (j = 0; j<kernel_dim[i][0]; j++)
				{
					local_res = 0.0;
					for (k = 0; k<kernel_dim[i][1]; k++)
					{
						local_res += local_input[k] * filters[i][0][0][k][j];
					}
					local_res = non_linear(0,local_res + bias[i][j]);
					inter_res[i + 1][0][0][j] = local_res;
				}
				free(local_input);				
			}
			else
			{
				for (j = 0; j<kernel_dim[i][0]; j++)
				{
					local_res = 0.0;
					for (k = 0; k<kernel_dim[i][1]; k++)
					{
						local_res += inter_res[i][0][0][k] * filters[i][0][0][k][j];
					}
					local_res = non_linear(0,local_res + bias[i][j]);
					inter_res[i + 1][0][0][j] = local_res;
				}
			}
			if(DEBUG_MODE)
			{
				printf("print result on FULL connent layer %d, it has %d outputs===========================================\n",i,kernel_dim[i][0]);
				for(j=0;j<kernel_dim[i][0];j++)
				{
					printf("%f ",inter_res[i + 1][0][0][j]);
				}
			}			
		}
	}

	if (DEBUG_MODE)
	{
		printf("execution done\n");
		printf("this net has %d output ports\n",kernel_dim[layer_num-1][0]);
		printf("the results are:\n\t");
		for (i = 0; i < kernel_dim[layer_num - 1][0]; i++)
		{
			printf("%f ", inter_res[layer_num][0][0][i]);
		}
		printf("\n");
	}
	
	float tmp_base = 0.0;
	int predict;
	for (i = 0; i < kernel_dim[layer_num - 1][0]; i++)
	{
		if (inter_res[layer_num][0][0][i] > tmp_base)
		{
			predict = i;
			tmp_base = inter_res[layer_num][0][0][i];
		}
	}
	printf("\nthe prediction is: %d\n", predict + 1);
	//also the reason for predict + 1 is the trained model in matlab's 10 output port represent 1 2 ....9 0 sequentially
	getchar();
	
	//recycle the mem
	//free mem store the internal result	
	for (i = 0; i < layer_num+1; i++)
	{
		for (j = 0; j < feature_num[i]; j++)
		{
				for (k = 0; k < inter_res_dim[i][1]; k++)
				{
					free(inter_res[i][j][k]);
				}
				free(inter_res[i][j]);
		}
		free(inter_res[i]);
		free(inter_res_dim[i]);
	}
	free(inter_res_dim);
	free(inter_res);



	//free the kernels
	for (i = 0; i < layer_num; i++)
	{
		if (filters[i] != NULL){
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
			free(bias[i]);
			free(filters[i]);
		}
		free(kernel_dim[i]);
	}
	free(filters);
	free(bias);
	free(kernel_dim);

	//free others
	free(layer_type);
	free(feature_num);
	free(method);
	free(stride);	
	return 0;
}

int main()
{
	run_cpu();
	return 0;
}
