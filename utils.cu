#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Global pointers for Merkle root
static BYTE *d_transactions = NULL;
static BYTE *d_hashes_A = NULL;
static BYTE *d_hashes_B = NULL;
static bool is_memory_allocated = false;

// Global pointers for Nonce finding
static BYTE *d_difficulty = NULL;
static BYTE *d_block_content = NULL;
static uint32_t *d_found_nonce = NULL;
static bool is_nonce_memory_allocated = false;

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    #pragma unroll
    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

// Kernel for computing the SHA256 hash for each transaction
__global__ void compute_hashes(BYTE *transactions, int transaction_size, int n, BYTE *hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Calculate the input and output pointers once
        const BYTE *in = transactions + idx * transaction_size;
        BYTE *out = hashes + idx * SHA256_HASH_SIZE;
        apply_sha256(in, out);
    }
}

// Kernel for combining two hashes and computing parent hash
__global__ void reduce_tree(BYTE *input_hashes, BYTE *output_hashes, int curr_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = (curr_n + 1) / 2;

    if (idx < num_pairs) {
        BYTE combined[129];
        int left_idx = 2 * idx;
        int right_idx = (2 * idx + 1 < curr_n) ? (2 * idx + 1) : left_idx;

        int left_offset = left_idx * SHA256_HASH_SIZE;
        int right_offset = right_idx * SHA256_HASH_SIZE;

        // Manual copy without d_strcpy/d_strcat to avoid null-terminator scanning
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            combined[i] = input_hashes[left_offset + i];
            combined[64 + i] = input_hashes[right_offset + i];
        }
        combined[128] = '\0';

        BYTE *out = output_hashes + idx * SHA256_HASH_SIZE;
        apply_sha256(combined, out);
    }
}

// TODO 1: Implement this function in CUDA
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    
    // Lazy memory allocation for GPU memory
    if (!is_memory_allocated) {
        cudaMalloc((void **)&d_transactions, max_transactions_in_a_block * transaction_size);
        cudaMalloc((void **)&d_hashes_A, max_transactions_in_a_block * SHA256_HASH_SIZE);
        cudaMalloc((void **)&d_hashes_B, max_transactions_in_a_block * SHA256_HASH_SIZE);
        is_memory_allocated = true;
    }

    // Copy input data to GPU memory
    cudaMemcpy(d_transactions, transactions, n * transaction_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    compute_hashes<<<blocks, threads_per_block>>>(d_transactions, transaction_size, n, d_hashes_A);
    cudaDeviceSynchronize();

    BYTE *d_in = d_hashes_A;
    BYTE *d_out = d_hashes_B;
    int curr_n = n;

    // Reduce the tree until we have a single hash
    while (curr_n > 1) {
        int num_pairs = (curr_n + 1) / 2;
        int blocks_reduce = (num_pairs + threads_per_block - 1) / threads_per_block;

        reduce_tree<<<blocks_reduce, threads_per_block>>>(d_in, d_out, curr_n);
        cudaDeviceSynchronize();

        BYTE *tmp = d_in;
        d_in = d_out;
        d_out = tmp;

        curr_n = num_pairs;
    }

    // Copy the result to the host
    cudaMemcpy(merkle_root, d_in, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
}

__global__ void find_nonce_kernel(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content_base, size_t current_length, uint32_t *d_found_nonce) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (uint64_t nonce = idx; nonce < max_nonce; nonce += stride) {
        
        // Volatile to ensure that the thread sees the immediate changes made by others in the memory
        if (*((volatile uint32_t *)d_found_nonce) != UINT32_MAX) {
            return;
        }

        // Local buffer for concatenation
        BYTE local_content[256]; 
        d_strcpy((char *)local_content, (const char *)block_content_base);

        // Convert nonce to string and concatenate to block content
        char nonce_string[NONCE_SIZE];
        intToString((uint32_t)nonce, nonce_string);
        d_strcpy((char *)local_content + current_length, nonce_string);

        // Compute hash of the block content with the nonce
        BYTE local_hash[SHA256_HASH_SIZE];
        apply_sha256(local_content, local_hash);

        // Compare hash with difficulty and update the nonce if found
        if (compare_hashes(local_hash, difficulty) <= 0) {
            atomicMin(d_found_nonce, (uint32_t)nonce);
        }
    }
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    
    // Lazy memory allocation for GPU memory
    if (!is_nonce_memory_allocated) {
        cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
        cudaMalloc((void **)&d_block_content, 256); 
        cudaMalloc((void **)&d_found_nonce, sizeof(uint32_t));
        is_nonce_memory_allocated = true;
    }

    // Copy input data to GPU memory
    cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_content, block_content, current_length + 1, cudaMemcpyHostToDevice);

    uint32_t reset_val = UINT32_MAX;
    cudaMemcpy(d_found_nonce, &reset_val, sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = 1024; 

    find_nonce_kernel<<<blocks, threads_per_block>>>(d_difficulty, max_nonce, d_block_content, current_length, d_found_nonce);
    cudaDeviceSynchronize();

    // Copy the found nonce to the host
    uint32_t h_found_nonce = UINT32_MAX; 
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // If a nonce was found, copy the block hash to the output and return 0
    if (h_found_nonce != UINT32_MAX) {
        *valid_nonce = h_found_nonce;
        
        // Final hash calculation on CPU to avoid complex operations on GPU
        char nonce_string[NONCE_SIZE];
        sprintf(nonce_string, "%u", h_found_nonce);
        strcpy((char *)block_content + current_length, nonce_string);
        apply_sha256(block_content, block_hash);
        
        return 0;
    }

    return 1;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
