#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


typedef __half22 newDT;

# define THREADS 32

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{

    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];

    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }

    // particles per celldt_sub_cycling
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;

    // cast it to required precision
    part->qom = (FPpart) param->qom[is];

    long npmax = part->npmax;

    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    //use pinned memory
    gpuCheck(cudaMallocHost(&part->x, sizeof(FPpart) * npmax));
    gpuCheck(cudaMallocHost(&part->y, sizeof(FPpart) * npmax));
    gpuCheck(cudaMallocHost(&part->z, sizeof(FPpart) * npmax));
    // allocate velocity
    gpuCheck(cudaMallocHost(&part->u, sizeof(FPpart) * npmax));
    gpuCheck(cudaMallocHost(&part->v, sizeof(FPpart) * npmax));
    gpuCheck(cudaMallocHost(&part->w, sizeof(FPpart) * npmax));
    // allocate charge = q * statistical weight
    gpuCheck(cudaMallocHost(&part->q, sizeof(FPinterp) * npmax));

}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    gpuCheck(cudaFreeHost(part->x));
    gpuCheck(cudaFreeHost(part->y));
    gpuCheck(cudaFreeHost(part->z));

    gpuCheck(cudaFreeHost(part->u));
    gpuCheck(cudaFreeHost(part->v));
    gpuCheck(cudaFreeHost(part->w));
    gpuCheck(cudaFreeHost(part->q));
}

/** GPU particle mover */
// zk: the main part of the GPU kenel is almost same as CPU program, so we add an input parameter idx, make this as a common kenel
// zk: both host and device can run this kernel
__global__ void mover_PC_gpu(particles* part_gpu, EMfield* field_gpu, grid* grid_gpu, parameters* param_gpu)
{

    // check();
    // printf("%f", (FPpart) param_gpu->dt/((double) part_gpu->n_sub_cycles));
    // this doesn't work in opt, the param_gpu->dt and so on does not exist
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= part_gpu->nop) {
      return;
    }

    // FPpart qom;
    newDT qomGPU = __float2half2_rn_rn2(part_gpu->qom);

    // // FPpart u0, v0, w0;
    // newDT u0GPU = __float2half(part_gpu->u0)
    // newDT v0GPU = __float2half(part_gpu->v0)
    // newDT w0GPU = __float2half(part_gpu->w0)
    // FPpart uth, vth, wth;
    // newDT uthGPU = __float2half(part_gpu->uth)
    // newDT vthGPU = __float2half(part_gpu->vth)
    // newDT wthGPU = __float2half(part_gpu->wth)

    newDT x_idx = __float2half2_rn2(part_gpu->x[idx]);
    newDT y_idx = __float2half2_rn2(part_gpu->y[idx]);
    newDT z_idx = __float2half2_rn2(part_gpu->z[idx]);
    newDT u_idx = __float2half2_rn2(part_gpu->u[idx]);
    newDT v_idx = __float2half2_rn2(part_gpu->v[idx]);
    newDT w_idx = __float2half2_rn2(part_gpu->w[idx]);
    // /** particle arrays: 1D arrays[npmax] */
    // FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    // /** q must have precision of interpolated quantities: typically double. Not used in mover */
    // FPinterp* q;


    // auxiliary variables
    // newDT dt_sub_cycling = (newDT) param_gpu->dt/((double) part_gpu->n_sub_cycles);
    newDT dt_sub_cycling = __double2half22(param_gpu->dt)/__int2hal2f2_rn(part_gpu->n_sub_cycles);

    newDT dto2 = __float2half2_rn2(0.5) * dt_sub_cycling;
    newDT qomdt2 = qomGPU*dto2/__double2half22(param_gpu->c);
    newDT omdtsq, denom, ut, vt, wt, udotb;
    // local (to the particle) electric and magnetic field_gpu
    newDT Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    newDT weight[2][2][2];
    newDT xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    newDT xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    //////////////////////////////////////////////
    // zk
    // n_sub_cycles is defined in input parameters, it is the subcycling with one time step dt
    // final update for location: x_idx = xptilde + uptilde*dt_sub_cycling; (dt_sub_cycling = param_gpu->dt / part_gpu->n_sub_cycles);)
    // this loop is for time marching, which can not be omitted
    //////////////////////////////////////////////
    // int stride;
    // if (idx >= 0) {
    //   stride = THREADS * (part_gpu -> nop + THREADS - 1) / THREADS;
    // }

    // start subcycling
    for (int i_sub=0; i_sub <  part_gpu->n_sub_cycles; i_sub++){

        ///////////////////////////////////////////
        // zk
        // this for loops for all particles, so we can modify this loop for GPU computing
        ///////////////////////////////////////////
        // for (int idx=0; idx <  part_gpu->nop; idx++){
        // for (int idx=idx; idx <  part_gpu->nop; idx+=stride){

            // move each particle with new fields

            xptilde = x_idx;
            yptilde = y_idx;
            zptilde = z_idx;
            ///////////////////////////////////////////
            // zk
            // NiterMover is defined by input parameter for electrons, and defined 1 for ions
            // x_idx will be renewed at the end of this loop, and read again at the next beginning
            // therefore, it is also a time dependent iteration
            ///////////////////////////////////////////

            // calculate the average velocity iteratively
            for(int innter=0; innter < part_gpu->NiterMover; innter++){
                // interpolation G-->P

                ix = 2 +  int((x_idx - __float2half2_rn2((grid_gpu->xStart)*grid_gpu->invdx)));
                iy = 2 +  int((y_idx - __float2half2_rn2((grid_gpu->yStart)*grid_gpu->invdy)));
                iz = 2 +  int((z_idx - __float2half2_rn2((grid_gpu->zStart)*grid_gpu->invdz)));

                // calculate weights
                ///////////////////////////////////////////
                // zk
                // We should insert idx in XN, YN, ZN, so we should use get_idx() function in alloc.h
                // We should use 1D array here, but the XN, YN, ZN are 3D array (generated in FPfield*** XN; in Grid.h)
                // So we should use XN_flat instead, which is an 1D array (generated in FPfield* XN_flat; in Grid.h)
                ///////////////////////////////////////////

                xi[0]   = x_idx - __float2half2_rn2(grid_gpu->XN_flat[get_idx(ix - 1, iy, iz, grid_gpu->nyn, grid_gpu->nzn)]);
                eta[0]  = y_idx - __float2half2_rn2(grid_gpu->YN_flat[get_idx(ix, iy - 1, iz, grid_gpu->nyn, grid_gpu->nzn)]);
                zeta[0] = z_idx - __float2half2_rn2(grid_gpu->ZN_flat[get_idx(ix, iy, iz - 1, grid_gpu->nyn, grid_gpu->nzn)]);
                xi[1]   = __float2half2_rn2(grid_gpu->XN_flat[get_idx(ix, iy, iz, grid_gpu->nyn, grid_gpu->nzn)]) - x_idx;
                eta[1]  = __float2half2_rn2(grid_gpu->YN_flat[get_idx(ix, iy, iz, grid_gpu->nyn, grid_gpu->nzn)]) - y_idx;
                zeta[1] = __float2half2_rn2(grid_gpu->ZN_flat[get_idx(ix, iy, iz, grid_gpu->nyn, grid_gpu->nzn)]) - z_idx;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = __float2half2_rn2(xi[ii]) *__float2half2_rnf2(eta[jj]) __float2half2_rnlf2(zeta[kk])__float2half2_rnalf2(grid_gpu->invVOL);

                // set to zero local electric and magnetic field_gpu
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                ///////////////////////////////////////////
                // zk
                // E and B are similar to XN, YN , ZN
                ///////////////////////////////////////////

                            Exl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                            Eyl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                            Ezl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                            Bxl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                            Byl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                            Bzl += weight[ii][jj][kk]*__float2half2_rn2(field_gpu->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grid_gpu->nyn, grid_gpu->nzn)]);
                        }

                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = __float2half2_rn2(1.0) / (__float2half2_rnf2(1.0) + omdtsq);
                // solve the position equation
                ut= u_idx + qomdt2*Exl;
                vt= v_idx + qomdt2*Eyl;
                wt= w_idx + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                x_idx = xptilde + uptilde*dto2;
                y_idx = yptilde + vptilde*dto2;
                z_idx = zptilde + wptilde*dto2;

            } // end of iteration

            // update the final position and velocity
            u_idx= __float2half2_rn2(2.0) * uptilde - u_idx;
            v_idx= __float2half2_rn2(2.0) * vptilde - v_idx;
            w_idx= __float2half2_rn2(2.0) * wptilde - w_idx;
            x_idx = xptilde + uptilde*dt_sub_cycling;
            y_idx = yptilde + vptilde*dt_sub_cycling;
            z_idx = zptilde + wptilde*dt_sub_cycling;


            //////////
            //////////
            ////////// BC

            part_gpu->x[idx] = __high2float2(x_idx);
            part_gpu->y[idx] = __high2float2(y_idx);
            part_gpu->z[idx] = __high2float2(z_idx);
            part_gpu->u[idx] = __high2float2(u_idx);
            part_gpu->v[idx] = __high2float2(v_idx);
            part_gpu->w[idx] = __high2float2(w_idx);

            // X-DIRECTION: BC particles
            if (part_gpu->x[idx] > grid_gpu->Lx){
                if (param_gpu->PERIODICX==true){ // PERIODIC
                    part_gpu->x[idx] = part_gpu->x[idx] - grid_gpu->Lx;
                } else { // REFLECTING BC
                    part_gpu->u[idx] = -part_gpu->u[idx];
                    part_gpu->x[idx] = 2*grid_gpu->Lx - part_gpu->x[idx];
                }
            }

            if (part_gpu->x[idx] < 0){
                if (param_gpu->PERIODICX==true){ // PERIODIC
                   part_gpu->x[idx] = part_gpu->x[idx] + grid_gpu->Lx;
                } else { // REFLECTING BC
                    part_gpu->u[idx] = -part_gpu->u[idx];
                    part_gpu->x[idx] = -part_gpu->x[idx];
                }
            }


            // Y-DIRECTION: BC particles
            if (part_gpu->y[idx] > grid_gpu->Ly){
                if (param_gpu->PERIODICY==true){ // PERIODIC
                    part_gpu->y[idx] = part_gpu->y[idx] - grid_gpu->Ly;
                } else { // REFLECTING BC
                    part_gpu->v[idx] = -part_gpu->v[idx];
                    part_gpu->y[idx] = 2*grid_gpu->Ly - part_gpu->y[idx];
                }
            }

            if (part_gpu->y[idx] < 0){
                if (param_gpu->PERIODICY==true){ // PERIODIC
                    part_gpu->y[idx] = part_gpu->y[idx] + grid_gpu->Ly;
                } else { // REFLECTING BC
                    part_gpu->v[idx] = -part_gpu->v[idx];
                    part_gpu->y[idx] = -part_gpu->y[idx];
                }
            }

            // Z-DIRECTION: BC particles
            if (part_gpu->z[idx] > grid_gpu->Lz){
                if (param_gpu->PERIODICZ==true){ // PERIODIC
                    part_gpu->z[idx] = part_gpu->z[idx] - grid_gpu->Lz;
                } else { // REFLECTING BC
                    part_gpu->w[idx] = -part_gpu->w[idx];
                    part_gpu->z[idx] = 2*grid_gpu->Lz - part_gpu->z[idx];
                }
            }

            if (part_gpu->z[idx] < 0){
                if (param_gpu->PERIODICZ==true){ // PERIODIC
                    part_gpu->z[idx] = part_gpu->z[idx] + grid_gpu->Lz;
                } else { // REFLECTING BC
                    part_gpu->w[idx] = -part_gpu->w[idx];
                    part_gpu->z[idx] = -part_gpu->z[idx];
                }
            }


        }  // end of subcycling

} // end of the mover

void pre_allocate(struct EMfield* field, struct EMfield* field_gpu, struct EMfield** field_gpu_ptr,
                struct grid* grd, struct grid* grid_gpu, struct grid** grid_gpu_ptr,
                struct parameters* param, struct parameters** param_gpu_ptr){
    // locate EMfield
    // struct field:  Ex, Ey, Ez, Bxn, Byn, Bzn are used, type: FPfield, size: total grid num * type size
    // only arrays in field are used
    size_t size_field_array = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield); //EMfield.cpp

    // allocate array memory in EMfield
    gpuCheck(cudaMalloc(&field_gpu->Ex_flat, size_field_array));
    gpuCheck(cudaMalloc(&field_gpu->Ey_flat, size_field_array));
    gpuCheck(cudaMalloc(&field_gpu->Ez_flat, size_field_array));
    gpuCheck(cudaMalloc(&field_gpu->Bxn_flat, size_field_array));
    gpuCheck(cudaMalloc(&field_gpu->Byn_flat, size_field_array));
    gpuCheck(cudaMalloc(&field_gpu->Bzn_flat, size_field_array));
    // copy array data from CPU to GPU
    gpuCheck(cudaMemcpy(field_gpu->Ex_flat, field->Ex_flat, size_field_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(field_gpu->Ey_flat, field->Ey_flat, size_field_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(field_gpu->Ez_flat, field->Ez_flat, size_field_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(field_gpu->Bxn_flat, field->Bxn_flat, size_field_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(field_gpu->Byn_flat, field->Byn_flat, size_field_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(field_gpu->Bzn_flat, field->Bzn_flat, size_field_array, cudaMemcpyHostToDevice));
    // allocate struct
    gpuCheck(cudaMalloc(field_gpu_ptr, sizeof(EMfield)));
    // bind pointer
    gpuCheck(cudaMemcpy(*field_gpu_ptr, field_gpu, sizeof(EMfield), cudaMemcpyHostToDevice));

    // locate grid
    // struct grd: XN, YN, ZN are use. type: FPfield, size: total grid num * type size
    // arrays and default data types are all used
    size_t size_grd_array = size_field_array;
    //  first we should copy the data of the default types to grid_gpu, now the grid_gpu is still on CPU
    memcpy(grid_gpu, grd, sizeof(grid));

    // allocate array memory in grid
    gpuCheck(cudaMalloc(&grid_gpu->XN_flat, size_grd_array));
    gpuCheck(cudaMalloc(&grid_gpu->YN_flat, size_grd_array));
    gpuCheck(cudaMalloc(&grid_gpu->ZN_flat, size_grd_array));
    // copy array data from CPU to GPU
    // now grid_gpu has pointers to arrays in GPU, and data of default type in CPU
    gpuCheck(cudaMemcpy(grid_gpu->XN_flat, grd->XN_flat, size_grd_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(grid_gpu->YN_flat, grd->YN_flat, size_grd_array, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(grid_gpu->ZN_flat, grd->ZN_flat, size_grd_array, cudaMemcpyHostToDevice));
    // allocate struct
    gpuCheck(cudaMalloc(grid_gpu_ptr, sizeof(grid)));
    // bind pointer and copy data of default type to GPU
    gpuCheck(cudaMemcpy(*grid_gpu_ptr, grid_gpu, sizeof(grid), cudaMemcpyHostToDevice));

    // struct param
    // param is much simple because no arrays are involved
    gpuCheck(cudaMalloc(param_gpu_ptr, sizeof(parameters)));
    gpuCheck(cudaMemcpy(*param_gpu_ptr, param, sizeof(parameters), cudaMemcpyHostToDevice));

  }


int mover_PC_gpu_launcher_opt(struct particles* part, struct particles* part_gpu, struct particles* part_gpu_ptr, struct EMfield* field_gpu_ptr, struct grid* grid_gpu_ptr, struct parameters* param_gpu_ptr){
  std::cout<<"run on GPU"<<std::endl;
  // define varable names

  // allocate GPU memory
  // we found that xptilde = part->x[idx] does not work.
  // what we copied above is just the pointer address of the struct, not the arry inside the struct
  // the arries must be allocated and copied manually

  // arrays and default data types are all used, so we mimic the method for grid
  size_t size_part = part -> npmax * sizeof(FPpart);

  memcpy(part_gpu, part, sizeof(particles));
  gpuCheck(cudaMalloc(&part_gpu->x, size_part));
  gpuCheck(cudaMalloc(&part_gpu->y, size_part));
  gpuCheck(cudaMalloc(&part_gpu->z, size_part));
  gpuCheck(cudaMalloc(&part_gpu->u, size_part));
  gpuCheck(cudaMalloc(&part_gpu->v, size_part));
  gpuCheck(cudaMalloc(&part_gpu->w, size_part));

  gpuCheck(cudaMemcpy(part_gpu->x, part->x, size_part, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(part_gpu->y, part->y, size_part, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(part_gpu->z, part->z, size_part, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(part_gpu->u, part->u, size_part, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(part_gpu->v, part->v, size_part, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(part_gpu->w, part->w, size_part, cudaMemcpyHostToDevice));
  // struct particles: x,y,z,u,v,w are used, type: FPpart, size: num particles*type size
  // linking pointer to arries
  gpuCheck(cudaMalloc(&part_gpu_ptr, sizeof(particles)));
  gpuCheck(cudaMemcpy(part_gpu_ptr, part_gpu, sizeof(particles), cudaMemcpyHostToDevice));

  // cuda dimension
  dim3 threads(THREADS, 1, 1);
  dim3 blocks((part -> nop + THREADS - 1) / THREADS, 1, 1);

  // launch kernel
  // print species and subcycling
  // std::cout << "***  MOVER with SUBCYCLYING "<< param_gpu->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
  mover_PC_gpu<<<blocks, threads>>>(part_gpu_ptr, field_gpu_ptr, grid_gpu_ptr, param_gpu_ptr);
  cudaDeviceSynchronize();

  // copy from device to host
  // the field is not updated, so we only need to copy paticles back to host
  // particle
  gpuCheck(cudaMemcpy(part->x, part_gpu->x, size_part, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(part->y, part_gpu->y, size_part, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(part->z, part_gpu->z, size_part, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(part->u, part_gpu->u, size_part, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(part->v, part_gpu->v, size_part, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(part->w, part_gpu->w, size_part, cudaMemcpyDeviceToHost));

  // free GPU memory
  gpuCheck(cudaFree(part_gpu_ptr));
  gpuCheck(cudaFree(part_gpu->x));
  gpuCheck(cudaFree(part_gpu->y));
  gpuCheck(cudaFree(part_gpu->z));
  gpuCheck(cudaFree(part_gpu->u));
  gpuCheck(cudaFree(part_gpu->v));
  gpuCheck(cudaFree(part_gpu->w));

  return 0;
}
int mover_PC_gpu_launcher_ori(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param){

    std::cout<<"run on GPU"<<std::endl;
    // define varable names
    particles *part_gpu;
    EMfield *field_gpu;
    grid *grid_gpu;
    parameters *param_gpu;

    // allocate GPU memory
    // we found that xptilde = part->x[idx] does not work.
    // what we copied above is just the pointer address of the struct, not the arry inside the struct
    // the arries must be allocated and copied manually

    // struct particles: x,y,z,u,v,w are used, type: FPpart, size: num particles*type size
    cudaMalloc((void**)&part_gpu, sizeof(particles));
    cudaMemcpy(part_gpu, part, sizeof(particles), cudaMemcpyHostToDevice);
    FPpart *x_gpu, *y_gpu, *z_gpu, *u_gpu, *v_gpu, *w_gpu;
    int size_part = part -> npmax * sizeof(FPpart);
    cudaMalloc((void**)&x_gpu, size_part);
    cudaMalloc((void**)&y_gpu, size_part);
    cudaMalloc((void**)&z_gpu, size_part);
    cudaMalloc((void**)&u_gpu, size_part);
    cudaMalloc((void**)&v_gpu, size_part);
    cudaMalloc((void**)&w_gpu, size_part);
    cudaMemcpy(x_gpu, part->x, size_part, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, part->y, size_part, cudaMemcpyHostToDevice);
    cudaMemcpy(z_gpu, part->z, size_part, cudaMemcpyHostToDevice);
    cudaMemcpy(u_gpu, part->u, size_part, cudaMemcpyHostToDevice);
    cudaMemcpy(v_gpu, part->v, size_part, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, part->w, size_part, cudaMemcpyHostToDevice);
    //linking pointer to arries
    // the pointer x_gpu is on host, so we should use cudaMemcpyHostToDevice
    cudaMemcpy(&(part_gpu->x), &x_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_gpu->y), &y_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_gpu->z), &z_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_gpu->u), &u_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_gpu->v), &v_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_gpu->w), &w_gpu, sizeof(FPpart*), cudaMemcpyHostToDevice);

    // struct field:  Ex, Ey, Ez, Bxn, Byn, Bzn are used, type: FPfield, size: total grid num * type size
    cudaMalloc(&field_gpu, sizeof(EMfield));
    cudaMemcpy(field_gpu, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    FPfield *Ex_gpu, *Ey_gpu, *Ez_gpu, *Bxn_gpu, *Byn_gpu, *Bzn_gpu;
    int size_field = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield); //EMfield.cpp
    cudaMalloc((void**)&Ex_gpu, size_field);
    cudaMalloc((void**)&Ey_gpu, size_field);
    cudaMalloc((void**)&Ez_gpu, size_field);
    cudaMalloc((void**)&Bxn_gpu, size_field);
    cudaMalloc((void**)&Byn_gpu, size_field);
    cudaMalloc((void**)&Bzn_gpu, size_field);
    cudaMemcpy(Ex_gpu, field->Ex_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_gpu, field->Ey_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_gpu, field->Ez_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_gpu, field->Bxn_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_gpu, field->Byn_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_gpu, field->Bzn_flat, size_field, cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Ex_flat), &Ex_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Ey_flat), &Ey_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Ez_flat), &Ez_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Bxn_flat), &Bxn_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Byn_flat), &Byn_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_gpu->Bzn_flat), &Bzn_gpu, sizeof(FPfield*), cudaMemcpyHostToDevice);

    //struct grd: XN, YN, ZN are use. type: FPfield, size: total grid num * type size
    cudaMalloc(&grid_gpu, sizeof(grid));
    cudaMemcpy(grid_gpu, grd, sizeof(grid), cudaMemcpyHostToDevice);
    FPfield *XN_gpu, *YN_gpu, *ZN_gpu;
    int size_grd = size_field;
    cudaMalloc((void**)&XN_gpu, size_grd);
    cudaMalloc((void**)&YN_gpu, size_grd);
    cudaMalloc((void**)&ZN_gpu, size_grd);
    cudaMemcpy(XN_gpu, grd->XN_flat, size_grd, cudaMemcpyHostToDevice);
    cudaMemcpy(YN_gpu, grd->YN_flat, size_grd, cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_gpu, grd->ZN_flat, size_grd, cudaMemcpyHostToDevice);
    cudaMemcpy(&(grid_gpu->XN_flat), &(XN_gpu), sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grid_gpu->YN_flat), &(YN_gpu), sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grid_gpu->ZN_flat), &(ZN_gpu), sizeof(FPfield*), cudaMemcpyHostToDevice);

    // struct param
    cudaMalloc((void**)&param_gpu, sizeof(parameters));
    cudaMemcpy(param_gpu, param, sizeof(parameters), cudaMemcpyHostToDevice);
    // cuda dimension
    dim3 threads(THREADS, 1, 1);
    dim3 blocks((part -> nop + THREADS - 1) / THREADS, 1, 1);

    // launch kernel
    // print species and subcycling
    // std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    mover_PC_gpu<<<blocks, threads>>>(part_gpu, field_gpu, grid_gpu, param_gpu);
    cudaDeviceSynchronize();

    // copy from device to host
    // we only need to copy paticles back to host
    // particle
    cudaMemcpy(part->x, x_gpu, size_part, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, y_gpu, size_part, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, z_gpu, size_part, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, u_gpu, size_part, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, v_gpu, size_part, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, w_gpu, size_part, cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(part_gpu);
    cudaFree(field_gpu);
    cudaFree(grid_gpu);
    cudaFree(param_gpu);

    return 0;
}

void gpu_deallocate(struct EMfield* field_gpu, struct EMfield** field_gpu_ptr,
                    struct grid* grid_gpu, struct grid** grid_gpu_ptr,
                    struct parameters** param_gpu_ptr){
    gpuCheck(cudaFree(*field_gpu_ptr));
    gpuCheck(cudaFree(field_gpu->Ex_flat));
    gpuCheck(cudaFree(field_gpu->Ey_flat));
    gpuCheck(cudaFree(field_gpu->Ez_flat));
    gpuCheck(cudaFree(field_gpu->Bxn_flat));
    gpuCheck(cudaFree(field_gpu->Byn_flat));
    gpuCheck(cudaFree(field_gpu->Bzn_flat));
    gpuCheck(cudaFree(*grid_gpu_ptr));
    gpuCheck(cudaFree(grid_gpu->XN_flat));
    gpuCheck(cudaFree(grid_gpu->YN_flat));
    gpuCheck(cudaFree(grid_gpu->ZN_flat));
    gpuCheck(cudaFree(*param_gpu_ptr));

}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // index of the cell
    int ix, iy, iz;


    for (register long long idx = 0; idx < part->nop; idx++) {

        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[idx] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[idx] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[idx] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0]   = part->x[idx] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[idx] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[idx] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[idx];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[idx];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[idx];

        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[idx] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[idx] * weight[ii][jj][kk];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;



        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[idx] * part->u[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[idx] * part->v[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;



        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[idx] * part->w[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[idx] * part->v[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[idx] * part->w[idx] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[idx] * part->w[idx] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    }

}