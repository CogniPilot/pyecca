/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) _ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fabs CASADI_PREFIX(fabs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_fabs(casadi_real x) {
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
  return x>0 ? x : -x;
#else
  return fabs(x);
#endif
}

static const casadi_int casadi_s0[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};

/* f_ref:(psi,psi_dot,psi_ddot,v_e[3],a_e[3],j_e[3],s_e[3],m,g,J_xx,J_yy,J_zz,J_xz)->(v_b[3],C_be[3x3],omega_eb_b[3],omega_dot_eb_b[3],M_b[3],T) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a3, a4, a5, a6, a7, a8, a9;
  a0=9.9999999999999995e-07;
  a1=arg[7]? arg[7][0] : 0;
  a2=arg[8]? arg[8][0] : 0;
  a3=arg[4]? arg[4][2] : 0;
  a2=(a2-a3);
  a2=(a1*a2);
  a3=arg[4]? arg[4][0] : 0;
  a3=(a1*a3);
  a4=casadi_sq(a3);
  a5=arg[4]? arg[4][1] : 0;
  a5=(a1*a5);
  a6=casadi_sq(a5);
  a4=(a4+a6);
  a6=casadi_sq(a2);
  a4=(a4+a6);
  a4=sqrt(a4);
  a6=(a0<a4);
  a4=(a6?a4:0);
  a6=(!a6);
  a6=(a6?a0:0);
  a4=(a4+a6);
  a2=(a2/a4);
  a6=arg[0]? arg[0][0] : 0;
  a7=sin(a6);
  a8=(a2*a7);
  a9=casadi_sq(a8);
  a6=cos(a6);
  a10=(a2*a6);
  a11=casadi_sq(a10);
  a9=(a9+a11);
  a5=(a5/a4);
  a11=(a5*a6);
  a3=(a3/a4);
  a12=(a3*a7);
  a11=(a11-a12);
  a12=casadi_sq(a11);
  a9=(a9+a12);
  a9=sqrt(a9);
  a12=(a0<a9);
  a10=(a10/a9);
  a10=(a12?a10:0);
  a13=(!a12);
  a14=1.;
  a13=(a13?a14:0);
  a10=(a10+a13);
  a13=(a10*a2);
  a11=(a11/a9);
  a14=(a11*a5);
  a14=(-a14);
  a14=(a12?a14:0);
  a13=(a13-a14);
  a14=arg[3]? arg[3][0] : 0;
  a15=(a13*a14);
  a16=(a11*a3);
  a16=(-a16);
  a16=(a12?a16:0);
  a8=(a8/a9);
  a9=(a8*a2);
  a9=(-a9);
  a9=(a12?a9:0);
  a16=(a16-a9);
  a9=arg[3]? arg[3][1] : 0;
  a17=(a16*a9);
  a15=(a15+a17);
  a17=(a8*a5);
  a17=(a12?a17:0);
  a18=(a10*a3);
  a17=(a17+a18);
  a18=arg[3]? arg[3][2] : 0;
  a19=(a17*a18);
  a15=(a15+a19);
  if (res[0]!=0) res[0][0]=a15;
  a15=(a8*a14);
  a15=(-a15);
  a15=(a12?a15:0);
  a19=(a10*a9);
  a15=(a15+a19);
  a19=(a11*a18);
  a19=(a12?a19:0);
  a15=(a15+a19);
  if (res[0]!=0) res[0][1]=a15;
  a18=(a2*a18);
  a14=(a3*a14);
  a9=(a5*a9);
  a14=(a14+a9);
  a18=(a18-a14);
  if (res[0]!=0) res[0][2]=a18;
  if (res[1]!=0) res[1][0]=a13;
  if (res[1]!=0) res[1][1]=a16;
  if (res[1]!=0) res[1][2]=a17;
  a18=(-a8);
  a18=(a12?a18:0);
  if (res[1]!=0) res[1][3]=a18;
  if (res[1]!=0) res[1][4]=a10;
  a18=(a12?a11:0);
  if (res[1]!=0) res[1][5]=a18;
  a18=(-a3);
  if (res[1]!=0) res[1][6]=a18;
  a18=(-a5);
  if (res[1]!=0) res[1][7]=a18;
  if (res[1]!=0) res[1][8]=a2;
  a14=(a1/a4);
  a9=arg[5]? arg[5][0] : 0;
  a15=(a14*a9);
  a19=(a15*a8);
  a19=(-a19);
  a19=(a12?a19:0);
  a20=arg[5]? arg[5][1] : 0;
  a21=(a14*a20);
  a22=(a21*a10);
  a19=(a19+a22);
  a22=arg[5]? arg[5][2] : 0;
  a14=(a14*a22);
  a23=(a14*a11);
  a23=(a12?a23:0);
  a19=(a19+a23);
  if (res[2]!=0) res[2][0]=a19;
  a15=(a15*a13);
  a21=(a21*a16);
  a15=(a15+a21);
  a14=(a14*a17);
  a15=(a15+a14);
  a14=(-a15);
  if (res[2]!=0) res[2][1]=a14;
  a14=asin(a3);
  a21=casadi_fabs(a14);
  a23=1.5707963267948966e+00;
  a21=(a21-a23);
  a21=casadi_fabs(a21);
  a21=(a21<a0);
  a21=(!a21);
  a18=atan2(a18,a2);
  a21=(a21?a18:0);
  a18=tan(a21);
  a18=(a15*a18);
  a23=cos(a14);
  a24=arg[1]? arg[1][0] : 0;
  a23=(a23*a24);
  a25=cos(a21);
  a26=casadi_fabs(a25);
  a0=(a0<a26);
  a0=(a0?a25:0);
  a23=(a23/a0);
  a18=(a18+a23);
  if (res[2]!=0) res[2][2]=a18;
  a23=(a1/a4);
  a0=arg[6]? arg[6][0] : 0;
  a25=(a0*a8);
  a25=(-a25);
  a25=(a12?a25:0);
  a26=arg[6]? arg[6][1] : 0;
  a27=(a26*a10);
  a25=(a25+a27);
  a27=arg[6]? arg[6][2] : 0;
  a28=(a27*a11);
  a28=(a12?a28:0);
  a25=(a25+a28);
  a23=(a23*a25);
  a25=2.;
  a22=(a1*a22);
  a22=(a22*a2);
  a9=(a1*a9);
  a9=(a9*a3);
  a20=(a1*a20);
  a20=(a20*a5);
  a9=(a9+a20);
  a22=(a22-a9);
  a25=(a25*a22);
  a25=(a25/a4);
  a22=(a25*a19);
  a23=(a23+a22);
  a22=(a18*a15);
  a23=(a23-a22);
  if (res[3]!=0) res[3][0]=a23;
  a1=(a1/a4);
  a0=(a0*a13);
  a26=(a26*a16);
  a0=(a0+a26);
  a27=(a27*a17);
  a0=(a0+a27);
  a1=(a1*a0);
  a25=(a25*a15);
  a1=(a1+a25);
  a25=(a18*a19);
  a1=(a1+a25);
  a25=(-a1);
  if (res[3]!=0) res[3][1]=a25;
  a25=arg[2]? arg[2][0] : 0;
  a0=(a13*a6);
  a27=(a7*a16);
  a0=(a0+a27);
  a27=(a13*a6);
  a26=(a7*a16);
  a27=(a27+a26);
  a0=(a0/a27);
  a26=(a0*a17);
  a22=(a6*a17);
  a22=(a22/a27);
  a9=(a22*a13);
  a20=(a7*a17);
  a20=(a20/a27);
  a27=(a20*a16);
  a9=(a9+a27);
  a26=(a26-a9);
  a26=(a26*a23);
  a25=(a25-a26);
  a26=(a22*a8);
  a26=(a12?a26:0);
  a9=(a20*a10);
  a26=(a26-a9);
  a9=(a0*a11);
  a9=(a12?a9:0);
  a26=(a26+a9);
  a26=(a26*a1);
  a25=(a25+a26);
  a26=(a16*a19);
  a10=(a10*a15);
  a26=(a26-a10);
  a10=(a5*a18);
  a26=(a26-a10);
  a10=sin(a14);
  a10=(a10*a24);
  a10=(a19+a10);
  a9=(a10*a17);
  a27=(a26*a9);
  a17=(a17*a19);
  a11=(a11*a15);
  a11=(-a11);
  a11=(a12?a11:0);
  a17=(a17+a11);
  a11=(a2*a18);
  a17=(a17+a11);
  a16=(a10*a16);
  a11=(a17*a16);
  a27=(a27-a11);
  a11=sin(a21);
  a14=cos(a14);
  a11=(a11*a14);
  a11=(a11*a24);
  a11=(a15+a11);
  a21=cos(a21);
  a11=(a11/a21);
  a6=(a11*a6);
  a6=(a24*a6);
  a27=(a27+a6);
  a27=(a22*a27);
  a10=(a10*a13);
  a17=(a17*a10);
  a13=(a13*a19);
  a8=(a8*a15);
  a12=(a12?a8:0);
  a13=(a13+a12);
  a12=(a3*a18);
  a13=(a13-a12);
  a9=(a13*a9);
  a17=(a17-a9);
  a11=(a11*a7);
  a24=(a24*a11);
  a17=(a17+a24);
  a17=(a20*a17);
  a27=(a27+a17);
  a13=(a13*a16);
  a26=(a26*a10);
  a13=(a13-a26);
  a13=(a0*a13);
  a27=(a27-a13);
  a25=(a25-a27);
  a22=(a22*a3);
  a20=(a20*a5);
  a22=(a22+a20);
  a0=(a0*a2);
  a22=(a22+a0);
  a25=(a25/a22);
  if (res[3]!=0) res[3][2]=a25;
  a22=arg[9]? arg[9][0] : 0;
  a0=(a22*a23);
  a2=arg[12]? arg[12][0] : 0;
  a20=(a2*a25);
  a0=(a0+a20);
  a20=arg[10]? arg[10][0] : 0;
  a5=(a20*a15);
  a3=(a18*a5);
  a27=(a2*a19);
  a13=arg[11]? arg[11][0] : 0;
  a26=(a13*a18);
  a27=(a27+a26);
  a26=(a15*a27);
  a3=(a3-a26);
  a0=(a0+a3);
  if (res[4]!=0) res[4][0]=a0;
  a22=(a22*a19);
  a0=(a2*a18);
  a22=(a22+a0);
  a18=(a18*a22);
  a27=(a19*a27);
  a18=(a18-a27);
  a20=(a20*a1);
  a18=(a18-a20);
  if (res[4]!=0) res[4][1]=a18;
  a2=(a2*a23);
  a13=(a13*a25);
  a2=(a2+a13);
  a15=(a15*a22);
  a19=(a19*a5);
  a15=(a15-a19);
  a2=(a2+a15);
  if (res[4]!=0) res[4][2]=a2;
  if (res[5]!=0) res[5][0]=a4;
  return 0;
}

int f_ref(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

int f_ref_alloc_mem(void) {
  return 0;
}

int f_ref_init_mem(int mem) {
  return 0;
}

void f_ref_free_mem(int mem) {
}

int f_ref_checkout(void) {
  return 0;
}

void f_ref_release(int mem) {
}

void f_ref_incref(void) {
}

void f_ref_decref(void) {
}

casadi_int f_ref_n_in(void) { return 13;}

casadi_int f_ref_n_out(void) { return 6;}

casadi_real f_ref_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

const char* f_ref_name_in(casadi_int i) {
  switch (i) {
    case 0: return "psi";
    case 1: return "psi_dot";
    case 2: return "psi_ddot";
    case 3: return "v_e";
    case 4: return "a_e";
    case 5: return "j_e";
    case 6: return "s_e";
    case 7: return "m";
    case 8: return "g";
    case 9: return "J_xx";
    case 10: return "J_yy";
    case 11: return "J_zz";
    case 12: return "J_xz";
    default: return 0;
  }
}

const char* f_ref_name_out(casadi_int i) {
  switch (i) {
    case 0: return "v_b";
    case 1: return "C_be";
    case 2: return "omega_eb_b";
    case 3: return "omega_dot_eb_b";
    case 4: return "M_b";
    case 5: return "T";
    default: return 0;
  }
}

const casadi_int* f_ref_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    case 3: return casadi_s1;
    case 4: return casadi_s1;
    case 5: return casadi_s1;
    case 6: return casadi_s1;
    case 7: return casadi_s0;
    case 8: return casadi_s0;
    case 9: return casadi_s0;
    case 10: return casadi_s0;
    case 11: return casadi_s0;
    case 12: return casadi_s0;
    default: return 0;
  }
}

const casadi_int* f_ref_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s2;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s1;
    case 5: return casadi_s0;
    default: return 0;
  }
}

int f_ref_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 13;
  if (sz_res) *sz_res = 6;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif