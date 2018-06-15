/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Loris Ercole, Federico Grasselli (SISSA)
   This version of Tersoff returns the same forces of pair tersoff, but
   implements the correct virial for many-body potentials, as explained in
     Fan et al., Phys. Rev. B 92, 094301 (2015)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_tersoff_fan.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairTersoffFan::PairTersoffFan(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  nelements = 0;
  elements = NULL;
  nparams = maxparam = 0;
  params = NULL;
  elem2param = NULL;
  map = NULL;

  maxshort = 10;
  neighshort = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTersoffFan::~PairTersoffFan()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(params);
  memory->destroy(elem2param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,evdwl,fpair;
  double rsq,rsq1,rsq2;
  double delr[3],delr1[3],delr2[3];
  double prefactor_ij;
  double fi_tmp[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = vflag_atom = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  const double cutshortsq = cutmax*cutmax;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // two-body interactions, skip half of them (0.5*d(fC_ij*fR_ij)/drij)

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delr[0] = x[j][0] - xtmp;   // delr = r_j - r_i = r_ij
      delr[1] = x[j][1] - ytmp;
      delr[2] = x[j][2] - ztmp;
      rsq = delr[0]*delr[0] + delr[1]*delr[1] + delr[2]*delr[2];

      // build a short neighbor list (up to max cutoff), for efficiency

      if (rsq < cutshortsq) {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < x[i][2]) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      jtype = map[type[j]];
      iparam_ij = elem2param[itype][jtype][jtype];
      if (rsq >= params[iparam_ij].cutsq) continue;

      repulsive(&params[iparam_ij],rsq,fpair,eflag,evdwl);

      // fpair does not have the 0.5 and has opposite sign

      f[i][0] -= delr[0]*fpair;
      f[i][1] -= delr[1]*fpair;
      f[i][2] -= delr[2]*fpair;
      f[j][0] += delr[0]*fpair;
      f[j][1] += delr[1]*fpair;
      f[j][2] += delr[2]*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delr[0],delr[1],delr[2]);
    }

    // compute and allocate zeta_ij for all the neighbors of i,
    // used to compute bij and bij_d

    double *zeta_matrix;
    zeta_matrix = new double[numshort];

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      iparam_ij = elem2param[itype][jtype][jtype];

      delr1[0] = x[j][0] - xtmp;   // delr1 = r_j - r_i = r_ij
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      if (rsq1 >= params[iparam_ij].cutsq) continue;

      // accumulate bondorder zeta for each i-j interaction via loop over k
      // zeta_ij computed and allocated only for the neighbors of i

      zeta_matrix[jj] = 0.0;
      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;   // delr2 = r_k - r_i = r_ik
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        zeta_matrix[jj] += zeta(&params[iparam_ijk],rsq1,rsq2,delr1,delr2);
      }
    }

    // loop over neighbors in the short list

    fi_tmp[0] = fi_tmp[1] = fi_tmp[2] = 0.0;
    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      iparam_ij = elem2param[itype][jtype][jtype];

      delr1[0] = x[j][0] - xtmp;   // delr1 = r_j - r_i = r_ij
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      if (rsq1 >= params[iparam_ij].cutsq) continue;

      // pairwise force due to zeta (0.5*b_ij*d(fA_ij*fC_ij)/drij)
      // returns prefactor_ij
 
      force_zeta(&params[iparam_ij],rsq1,zeta_matrix[jj],fpair,prefactor_ij,
                 eflag,evdwl);

      fi_tmp[0] = -delr1[0]*fpair;
      fi_tmp[1] = -delr1[1]*fpair;
      fi_tmp[2] = -delr1[2]*fpair;

      f[i][0] -= fi_tmp[0];
      f[i][1] -= fi_tmp[1];
      f[i][2] -= fi_tmp[2];
      f[j][0] += fi_tmp[0];
      f[j][1] += fi_tmp[1];
      f[j][2] += fi_tmp[2];

      if (evflag) ev_tally_manybody(i,j,nlocal,newton_pair,evdwl,0.0,
                                    fi_tmp,delr1);

      // attractive term via loop over k
 
      fi_tmp[0] = fi_tmp[1] = fi_tmp[2] = 0.0;
      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;   // delr2 = r_k - r_i = r_ik
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        if (rsq2 >= params[iparam_ijk].cutsq) continue;
 
        double ftmp[3] = {0.};

        attractive_fan(&params[iparam_ij],&params[iparam_ijk],prefactor_ij,
                       zeta_matrix[jj],zeta_matrix[kk],rsq1,rsq2,delr1,delr2,
                       ftmp);

        fi_tmp[0] -= ftmp[0];
        fi_tmp[1] -= ftmp[1];
        fi_tmp[2] -= ftmp[2];
      }

      f[i][0] -= fi_tmp[0];
      f[i][1] -= fi_tmp[1];
      f[i][2] -= fi_tmp[2];
      f[j][0] += fi_tmp[0];
      f[j][1] += fi_tmp[1];
      f[j][2] += fi_tmp[2];

      if (vflag_atom) ev_tally_manybody(i,j,nlocal,newton_pair,0.0,0.0,
                                        fi_tmp,delr1);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTersoffFan::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTersoffFan::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTersoffFan::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style TersoffFan requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style TersoffFan requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTersoffFan::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::read_file(char *file)
{
  int params_per_line = 17;
  char **words = new char*[params_per_line+1];

  memory->sfree(params);
  params = NULL;
  nparams = maxparam = 0;

  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open TersoffFan potential file %s",file);
      error->one(FLERR,str);
    }
  }

  // read each line out of file, skipping blank lines or leading '#'
  // store line of params if all 3 element tags are in element list

  int n,nwords,ielement,jelement,kelement;
  char line[MAXLINE],*ptr;
  int eof = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // concatenate additional lines until have params_per_line words

    while (nwords < params_per_line) {
      n = strlen(line);
      if (comm->me == 0) {
        ptr = fgets(&line[n],MAXLINE-n,fp);
        if (ptr == NULL) {
          eof = 1;
          fclose(fp);
        } else n = strlen(line) + 1;
      }
      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof) break;
      MPI_Bcast(&n,1,MPI_INT,0,world);
      MPI_Bcast(line,n,MPI_CHAR,0,world);
      if ((ptr = strchr(line,'#'))) *ptr = '\0';
      nwords = atom->count_words(line);
    }

    if (nwords != params_per_line)
      error->all(FLERR,"Incorrect format in TersoffFan potential file");

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    // ielement,jelement,kelement = 1st args
    // if all 3 args are in element list, then parse this line
    // else skip to next line

    for (ielement = 0; ielement < nelements; ielement++)
      if (strcmp(words[0],elements[ielement]) == 0) break;
    if (ielement == nelements) continue;
    for (jelement = 0; jelement < nelements; jelement++)
      if (strcmp(words[1],elements[jelement]) == 0) break;
    if (jelement == nelements) continue;
    for (kelement = 0; kelement < nelements; kelement++)
      if (strcmp(words[2],elements[kelement]) == 0) break;
    if (kelement == nelements) continue;

    // load up parameter settings and error check their values

    if (nparams == maxparam) {
      maxparam += DELTA;
      params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                          "pair:params");
    }

    params[nparams].ielement = ielement;
    params[nparams].jelement = jelement;
    params[nparams].kelement = kelement;
    params[nparams].powerm = atof(words[3]);
    params[nparams].gamma = atof(words[4]);
    params[nparams].lam3 = atof(words[5]);
    params[nparams].c = atof(words[6]);
    params[nparams].d = atof(words[7]);
    params[nparams].h = atof(words[8]);
    params[nparams].powern = atof(words[9]);
    params[nparams].beta = atof(words[10]);
    params[nparams].lam2 = atof(words[11]);
    params[nparams].bigb = atof(words[12]);
    params[nparams].bigr = atof(words[13]);
    params[nparams].bigd = atof(words[14]);
    params[nparams].lam1 = atof(words[15]);
    params[nparams].biga = atof(words[16]);

    // currently only allow m exponent of 1 or 3

    params[nparams].powermint = int(params[nparams].powerm);

    if (params[nparams].c < 0.0 ||
        params[nparams].d < 0.0 ||
        params[nparams].powern < 0.0 ||
        params[nparams].beta < 0.0 ||
        params[nparams].lam2 < 0.0 ||
        params[nparams].bigb < 0.0 ||
        params[nparams].bigr < 0.0 ||
        params[nparams].bigd < 0.0 ||
        params[nparams].bigd > params[nparams].bigr ||
        params[nparams].lam1 < 0.0 ||
        params[nparams].biga < 0.0 ||
        params[nparams].powerm - params[nparams].powermint != 0.0 ||
        (params[nparams].powermint != 3 &&
         params[nparams].powermint != 1) ||
        params[nparams].gamma < 0.0)
      error->all(FLERR,"Illegal TersoffFan parameter");

    nparams++;
  }

  delete [] words;
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::setup_params()
{
  int i,j,k,m,n;

  // set elem2param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,nelements,nelements,"pair:elem2param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2param[i][j][k] = n;
      }


  // compute parameter values derived from inputs

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].bigr + params[m].bigd;
    params[m].cutsq = params[m].cut*params[m].cut;

    params[m].c1 = pow(2.0*params[m].powern*1.0e-16,-1.0/params[m].powern);
    params[m].c2 = pow(2.0*params[m].powern*1.0e-8,-1.0/params[m].powern);
    params[m].c3 = 1.0/params[m].c2;
    params[m].c4 = 1.0/params[m].c1;
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++)
    if (params[m].cut > cutmax) cutmax = params[m].cut;
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::repulsive(Param *param, double rsq, double &fforce,
                            int eflag, double &eng)
{
  double r,tmp_fc,tmp_fc_d,tmp_exp;

  r = sqrt(rsq);
  tmp_fc = ters_fc(r,param);
  tmp_fc_d = ters_fc_d(r,param);
  tmp_exp = exp(-param->lam1 * r);
  fforce = -param->biga * tmp_exp * (tmp_fc_d - tmp_fc*param->lam1) / r;
  if (eflag) eng = tmp_fc * param->biga * tmp_exp;
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::zeta(Param *param, double rsqij, double rsqik,
                         double *delrij, double *delrik)
{
  double rij,rik,costheta,arg,ex_delr;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  costheta = (delrij[0]*delrik[0] + delrij[1]*delrik[1] +
              delrij[2]*delrik[2]) / (rij*rik);

  if (param->powermint == 3) arg = pow(param->lam3 * (rij-rik),3.0);
  else arg = param->lam3 * (rij-rik);

  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);

  return ters_fc(rik,param) * ters_gijk(costheta,param) * ex_delr;
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::force_zeta(Param *param, double rsq, double zeta_ij,
                             double &fforce, double &prefactor,
                             int eflag, double &eng)
{
  double r,fa,fa_d,bij,bij_d;

  r = sqrt(rsq);
  fa = ters_fa(r,param);  // f_A(r) * f_C(r)
  fa_d = ters_fa_d(r,param);
  bij = ters_bij(zeta_ij,param);
  bij_d = ters_bij_d(zeta_ij,param);

  fforce = 0.5*bij*fa_d / r;
  prefactor = 0.5*fa*bij_d;  // sign changed for consistency!
  if (eflag) eng = 0.5*bij*fa;
}

/* ----------------------------------------------------------------------
   attractive term
   use param_ij cutoff for rij test
   use param_ijk cutoff for rik test
------------------------------------------------------------------------- */

void PairTersoffFan::attractive_fan(Param *param_ij, Param *param_ijk, 
                                    double prefactor_ij, double zeta_ij,
                                    double zeta_ik, double rsqij, double rsqik,
                                    double *delrij, double *delrik, double *fi)
{
  double rij_hat[3],rik_hat[3];
  double rij,rijinv,rik,rikinv;

  rij = sqrt(rsqij);
  rijinv = 1.0/rij;
  vec3_scale(rijinv,delrij,rij_hat);  // rij_hat

  rik = sqrt(rsqik);
  rikinv = 1.0/rik;
  vec3_scale(rikinv,delrik,rik_hat);  // rik_hat

  ters_zetaterm_d_fan(prefactor_ij,zeta_ij,zeta_ik,rij_hat,rij,rik_hat,rik,fi,
                      param_ij,param_ijk);
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_fc(double r, Param *param)
{
  double ters_R = param->bigr;
  double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 1.0;
  if (r > ters_R+ters_D) return 0.0;
  return 0.5*(1.0 - sin(MY_PI2*(r - ters_R)/ters_D));
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_fc_d(double r, Param *param)
{
  double ters_R = param->bigr;
  double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 0.0;
  if (r > ters_R+ters_D) return 0.0;
  return -(MY_PI4/ters_D) * cos(MY_PI2*(r - ters_R)/ters_D);
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r) * ters_fc(r,param);
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) *
    (param->lam2 * ters_fc(r,param) - ters_fc_d(r,param));
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_bij(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  if (tmp > param->c1) return 1.0/sqrt(tmp);
  if (tmp > param->c2)
    return (1.0 - pow(tmp,-param->powern) / (2.0*param->powern))/sqrt(tmp);
  if (tmp < param->c4) return 1.0;
  if (tmp < param->c3)
    return 1.0 - pow(tmp,param->powern)/(2.0*param->powern);
  return pow(1.0 + pow(tmp,param->powern), -1.0/(2.0*param->powern));
}

/* ---------------------------------------------------------------------- */

double PairTersoffFan::ters_bij_d(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  if (tmp > param->c1) return param->beta * -0.5*pow(tmp,-1.5);
  if (tmp > param->c2)
    return param->beta * (-0.5*pow(tmp,-1.5) *
                          // error in negligible 2nd term fixed 9/30/2015
                          // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                          (1.0 - (1.0 +  1.0/(2.0*param->powern)) *
                           pow(tmp,-param->powern)));
  if (tmp < param->c4) return 0.0;
  if (tmp < param->c3)
    return -0.5*param->beta * pow(tmp,param->powern-1.0);

  double tmp_n = pow(tmp,param->powern);
  return -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*param->powern)))*tmp_n / zeta;
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::ters_zetaterm_d_fan(double prefactor_ij,
                                         double zeta_ij, double zeta_ik,
                                         double *rij_hat, double rij,
                                         double *rik_hat, double rik,
                                         double *drj,
                                         Param *param_ij, Param *param)
{
  // we are going to compute drj, i.e. the terms of dU_i/dr_ij containing
  // the derivatives of zeta_ij and zeta_ik
  double gijk,gijk_d,ex_delr,ex_delr_d,cos_theta,tmp;
  double fc_ik,fcfa_ik,fc_ij,fc_d_ij,bik_d,prefactor_ik;
  double ex_delr_neg,ex_delr_d_neg;
  double dcosdrij[3], drj_tmp[3];

  // ** actually the terms ij (fc_ij, fc_d_ij) are equal for all k's **

  fc_ik = ters_fc(rik,param); 

  // the following four parameters are needed in the additional term (VERDE)
  fcfa_ik = ters_fa(rik,param);
  bik_d   = ters_bij_d(zeta_ik,param);
  prefactor_ik = 0.5 * fcfa_ik * bik_d;

  fc_ij   = ters_fc(rij,param_ij);
  fc_d_ij = ters_fc_d(rij,param_ij);

  if (param->powermint == 3) tmp = pow(param->lam3 * (rij-rik),3.0);
  else tmp = param->lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (param->powermint == 3)
    ex_delr_d = 3.0*pow(param->lam3,3.0) * pow(rij-rik,2.0)*ex_delr;
  else ex_delr_d = param->lam3 * ex_delr;
 
  // when j -> k , then (rij-rik) -> (rik-rij) = -(rij-rik)
  // hence the corresponding exponential factor shall be 
  // for both lam3 == 1 and == 3
  if (-tmp > 69.0776) ex_delr_neg = 1.e30;
  else if (-tmp < -69.0776) ex_delr_neg = 0.0;
  else ex_delr_neg = exp(-tmp);

  // and its derivative shall be
  if (param->powermint == 3)
    ex_delr_d_neg = -3.0*pow(param->lam3,3.0) * pow(rik-rij,2.0)*ex_delr_neg;
  else ex_delr_d_neg = -param->lam3 * ex_delr_neg;

  cos_theta = vec3_dot(rij_hat,rik_hat);
  gijk = ters_gijk(cos_theta,param);
  gijk_d = ters_gijk_d(cos_theta,param);
 
  costheta_d_fan(rij_hat,rij,rik_hat,rik,dcosdrij);

  // compute the derivative of zetaterm_ij wrt vec{r}_{ij}:
  // drj  = (fc_ik * gijk_d * ex_delr * dcosdrij
  //         + fc_ik * gijk * ex_delr_d * rij_hat) * prefactor_ij

  vec3_scale(fc_ik* gijk_d * ex_delr,dcosdrij,drj);
  vec3_scaleadd(fc_ik * gijk * ex_delr_d,rij_hat,drj,drj);
  vec3_scale(prefactor_ij,drj,drj);

  // computing the derivative of zetaterm_ik wrt vec{r}_{ij}, 
  // drj_tmp  = (fc_d_ij * gijk * ex_delr_neg * rij_hat
  //             + fc_ij * gijk_d * ex_delr_neg * dcosdrij
  //             + fc_ij * gijk * ex_delr_d_neg * rij_hat) * prefactor_ik

  vec3_scale(fc_d_ij * gijk * ex_delr_neg,rij_hat,drj_tmp);
  vec3_scaleadd(fc_ij * gijk_d * ex_delr_neg,dcosdrij,drj_tmp,drj_tmp);
  vec3_scaleadd(fc_ij * gijk * ex_delr_d_neg,rij_hat,drj_tmp,drj_tmp);
  vec3_scale(prefactor_ik,drj_tmp,drj_tmp);

  // add the two contributions
  vec3_add(drj_tmp,drj,drj);
}

/* ---------------------------------------------------------------------- */

void PairTersoffFan::costheta_d_fan(double *rij_hat, double rij,
                                    double *rik_hat, double rik,
                                    double *dcosdrij)
{
  // derivative of cos(theta_ijk) wrt \vec{r}_{ij}
  //                = (-cos(theta_ijk)*rij_hat + rik_hat)/rij
  // note that theta_ijk = theta_ikj

  double cos_theta = vec3_dot(rij_hat,rik_hat);

  vec3_scaleadd(-cos_theta,rij_hat,rik_hat,dcosdrij);
  vec3_scale(1.0/rij,dcosdrij,dcosdrij);
}
