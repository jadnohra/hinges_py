from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy
import math
import copy
import time
import os, sys
try:
	import fcntl
except ImportError:
	fcntl = None

def dep_path(fname):
	return os.path.join(os.path.dirname(os.path.realpath(__file__)), fname)
execfile(dep_path('helper_arg.py'))
execfile(dep_path('helper_math1.py'))


################################################################################
# NP Dynamics
################################################################################
def np_create(mass):
	np = { 'iM':0.0, 'q':v3_z(), 'v':v3_z() }
	np['iM'] = 1.0/mass
	return np

def np_xfm(rb):
	M = mat_id(); mat_set_transl(M, np['q'][:3]); return M;

def np_draw(np, col_func):
	np_col = np.get('col', [1.0, 1.0, 1.0]); col_func(np_col[0], np_col[1], np_col[2]);
	M = np_xfm(rb)
	scene_xfm(M)
	scene_draw_box(2.0)

################################################################################
# RB Dynamics
################################################################################
def rb_create():
	rb = { 'iM':[0.0]*4, 'q':v3_z()+uquat_id(), 'v':[0.0]*6 }
	if arg_has('-fwd_euler'):
		rb['pv'] = rb['v']
	return rb

def rb_create_box(dims, mass):
	rb = rb_create()
	rb['dims'] = dims
	if (mass != float('inf')):
		rb['iM'][0] = 1.0/mass
		rb['iM'][1+0] = 12.0/(mass*(m_sq(dims[1])+m_sq(dims[2])))
		rb['iM'][1+1] = 12.0/(mass*(m_sq(dims[0])+m_sq(dims[2])))
		rb['iM'][1+2] = 12.0/(mass*(m_sq(dims[0])+m_sq(dims[1])))
	return rb

def rb_xfm(rb):
	M = mat_id(); mat_set_transl(M, rb['q'][:3]); mat_set_rot(M, uquat_to_mat(rb['q'][3:])); return M;
def rb_xfm_rot(rb):
	return uquat_to_mat(rb['q'][3:])
def rb_pt_local(rb, wpt):
	return mat4_mul_pt(mat_rigid_inv(rb_xfm(rb)), wpt)
def rb_vec_local(rb, wvec):
	return mat_mul_vec(mat_transp(rb_xfm_rot(rb)), wvec)
def rb_pt_local2(rb, pt, frame):
	return rb_pt_local(rb, pt) if frame=='w' else pt
def rb_vec_local2(rb, vec, frame):
	return rb_vec_local(rb, vec) if frame=='w' else vec
def rb_pt_world(rb, lpt):
	return mat4_mul_pt(rb_xfm(rb), lpt)
def rb_pt_world2(rb, pt, frame):
	return rb_pt_world(rb, pt) if frame =='l' else pt
def rb_vec_world(rb, lvec):
	return mat4_mul_vec(rb_xfm(rb), lvec)
def rb_vec_world2(rb, vec, frame):
	return rb_vec_world(rb, vec) if frame =='l' else vec

def rb_draw(rb, col_func):
	rb_col = rb.get('col', [1.0, 1.0, 1.0]); col_func(rb_col[0], rb_col[1], rb_col[2]);
	M = rb_xfm(rb)
	scene_xfm(M)
	scene_draw_box(rb['dims'])
	if False and rb['iM'][0] != 0.0:
		scene_draw_cs([1.0/x for x in rb['iM'][1:]], col_func)
	if False:
		col_func(1.0,0.0,0.0)
		scene_draw_box([3.0*rb['v'][3:][0],0.0,0.0])
		col_func(0.0,1.0,0.0)
		scene_draw_box([0.0,3.0*rb['v'][3:][1],0.0])
		col_func(0.0,0.0,1.0)
		scene_draw_box([0.0,0.0,3.0*rb['v'][3:][2]])

def rb_spherical(rbs, bi1, bi2, (pt1, frame1), (pt2, frame2)):
	b1,b2 = [rbs['bodies'][x] for x in (bi1, bi2)]
	sph = { 'type':'sph', 'bi':(bi1, bi2), 'larm':(rb_pt_local2(b1, pt1, frame1), rb_pt_local2(b2, pt2, frame2)) }
	return sph

def rb_hinge(rbs, bi1, bi2, (pt1_1, frame1_1), (axis_1, frame1_2), (pt2, frame2), (axis_2, frame2_2)):
	sph1 = rb_spherical(rbs, bi1, bi2, (pt1_1, frame1_1), (pt2, frame2))
	b1,b2 = [rbs['bodies'][x] for x in (bi1, bi2)]
	wpt_axis1 = rb_pt_world(b1, vec_add(sph1['larm'][0], rb_vec_local2(b1, axis_1, frame1_2)))
	wpt_axis2 = rb_pt_world(b2, vec_add(sph1['larm'][1], rb_vec_local2(b2, axis_2, frame2_2)))
	sph2 = rb_spherical(rbs, bi1, bi2, (wpt_axis1, 'w'), (wpt_axis2, 'w'))
	hng = {'type':'batch', 'bi':(bi1, bi2), 'batch':(sph1, sph2) }
	return hng

def rb_gamma_ls_block_solve(J,iMl,v,bias):
	gamma = mat_mul(mat_mul_diag(J, iMl), mat_transp(J))
	ig = mat_inv(gamma)
	sol = mat_mul_vec(ig, vec_sub(bias, mat_mul_vec(J,v)))
	return sol
def rb_gather_iMl(bodies):
	iMl = []
	for b in bodies:
		iMl.extend( [b['iM'][0]]*3 + b['iM'][1:] )
	return iMl
def rb_gather_v(bodies):
	v = []
	for b in bodies:
		v.extend( b['v'] )
	return v

def rb_step_si_sph(sph, bodies, h, stats):
	(b1,b2) = bodies
	J = []; bias = []; iMl = rb_gather_iMl(bodies); v = rb_gather_v(bodies);
	arms = rb_vec_world(b1, sph['larm'][0]), rb_vec_world(b2, sph['larm'][1])
	verr = vec_sub(rb_pt_world(b2, sph['larm'][1]), rb_pt_world(b1, sph['larm'][0]))
	stats['err_sq'] = stats['err_sq'] + m_sq(vec_norm(verr))
	for di in range(3):
		Ji = [v3_z(),v3_z(),v3_z(),v3_z()];
		Ji[0][di] = 1.0; Ji[1] = v3_cross(arms[0], Ji[0]).tolist();
		Ji[2+0][di] = -1.0; Ji[2+1] = v3_cross(arms[1], Ji[2+0]).tolist();
		for ai in range(2): # Turn to 'local' Jacobian counterpart
			Ji[2*ai+1] = rb_vec_local(bodies[ai], Ji[2*ai+1]).tolist()
		J.append(reduce(lambda x,y:x+y, Ji));
		bias.append( float(arg_get('-baumg', 0.8)) * (vec_dot(Ji[0], verr) / h) )
	lambdas = rb_gamma_ls_block_solve(J, iMl, v, bias)
	imps = mat_mul_diag(mat_mul_vec(mat_transp(J), lambdas), iMl)
	nv = vec_add(v, imps)
	b1['v'] = nv[:6]; b2['v'] = nv[6:];

def rb_sph_draw(sph, bds, col_func):
	coms = [rb_pt_world(bds[i], v3_z()) for i in range(2)]
	wpts = [rb_pt_world(bds[i], sph['larm'][i]) for i in range(2)]
	cols = (col_grn, col_bl, col_rd)
	scene_xfm_id()
	for i in range(2):
		scene_draw_line(coms[i], wpts[i], cols[i], col_func)
	scene_draw_line(wpts[0], wpts[1], cols[2], col_func)

def rb_pt_anim(rbs, bi, sample_func, sample_func_state):
	return {'bi':bi, 't':0, 'sample':sample_func, 'sample_state':sample_func_state }

def rb_anim_sph_sample_func_state(r, speed, xfm=mat_id()):
	return {'coords':[0.0,math.pi/2.0,r], 'speed':speed, 'xfm':xfm}

def rb_anim_sph_sample_func(t, h, state):
	crds = state['coords']
	crds[0] = t*state['speed']; pt = mat4_mul_pt(state['xfm'], coord_sph_to_cart2(crds));
	pt2 = mat4_mul_pt(state['xfm'], coord_sph_to_cart2(((t+h)*state['speed'], crds[1], crds[2])))
	return pt, vec_muls(vec_sub(pt2, pt), 1.0/h)

def rb_get_step_q(rb, h, v):
	q = copy.copy(rb['q'])
	q[:3] = vec_add(q[:3], vec_muls(v[:3], h))
	q[3:] = quat_mul(q[3:], rv_to_uquat(vec_muls(v[3:],h)) )
	q[3:] = vec_normd(q[3:])
	return q

def rb_step_q(rb, h, v):
	rb['q'] = rb_get_step_q(rb, h, v)

def rbs_create():
	return { 'bodies':[], 'constraints':[], 'anims':[], 'g':v3_z(), 'stats':{'err_sq':0.0} }

def rbs_get_body(rbs, rbi):
	return rbs['bodies'][rbi]

def rbs_add_body(rbs, rb):
	rbs['bodies'].append(rb); return len(rbs['bodies'])-1;

def rbs_find_body_index(rbs, value, field = 'name'):
	for bi in range(len(rbs['bodies'])):
		if (rbs['bodies'][bi][field] == value):
			return bi
	return -1

def rbs_find_bodies(rbs, values, field = 'name'):
	return [rbs['bodies'][bi] for bi in [rbs_find_body_index(rbs, vi, field) for vi in values]]

def rbs_add_constraint(rbs, ctrt):
	rbs['constraints'].append(ctrt)

def rbs_remove_constraint(rbs, ctrt):
	rbs['constraints'].remove(ctrt)

def rbs_add_anim(rbs, anim):
	rbs['anims'].append(anim)

def rbs_step_anims(rbs, h):
	for anim in rbs['anims']:
		t = anim['t']
		q,v = anim['sample'](t, h, anim['sample_state'])
		anim['t'] = t+h
		b = rbs['bodies'][anim['bi']]
		b['q'][:3] = q; b['v'][:3] = v;

def rbs_step_si_v(rbs, h, iters):
	stats = rbs['stats']; stats['err_sq'] = 0.0;
	def step_ctr(ctr, h):
		b1,b2 = [bodies[x] for x in ctr['bi']]
		if (ctr['type'] == 'sph'):
			rb_step_si_sph(ctr, (b1,b2), h, stats)
		elif (ctr['type'] == 'batch'):
			for bctr in ctr['batch']:
				step_ctr(bctr, h)
	bodies = rbs['bodies']
	if (rbs['g'] != v3_z()):
		g = rbs['g']; dv = vec_muls(g,h);
		for rb in rbs['bodies']:
			if (rb['iM'][0] != 0.0):
				rb['v'][:3] = vec_add(rb['v'][:3], dv)

	for it in range(iters):
		stats['err_sq'] = 0.0;
		for ctr in rbs['constraints']:
			step_ctr(ctr, h)

def rbs_step_q(rbs, h):
	vkey = 'pv' if arg_has('-fwd_euler') else 'v'
	for rb in rbs['bodies']:
		rb_step_q(rb, h, rb[vkey])

def rbs_v_to_pv(rbs):
	for rb in rbs['bodies']:
		rb['pv'] = copy.copy(rb['v'])

def rbs_step(rbs, h, si_iters):
	if arg_has('-fwd_euler'):
		rbs_v_to_pv(rbs)
	rbs_step_anims(rbs, h)
	rbs_step_si_v(rbs, h, si_iters)
	rbs_step_q(rbs, h)

def rbs_draw(rbs, col_func):
	def draw_ctr(ctr):
		b1,b2 = [rbs['bodies'][x] for x in ctr['bi']]
		if (ctr['type'] == 'sph'):
			rb_sph_draw(ctr, (b1,b2), col_func)
		elif (ctr['type'] == 'batch'):
			for bctr in ctr['batch']:
				draw_ctr(bctr)

	for rb in rbs['bodies']:
		rb_draw(rb, col_func)
	if False:
		for ctr in rbs['constraints']:
			draw_ctr(ctr)

################################################################################
# Scenes
################################################################################

def scene_1_update(sctx):
	scene = sctx['scene']
	opt_ctr = str(arg_get('-ctr', '1,2,3')).split(',')
	opt_wobble = str(arg_get('-wobble', '1,2,3')).split(',')
	opt_mass1 = float(arg_get('-mass1', 'inf'))
	if (sctx['frame'] == 0):
		scene['rbs'] = rbs_create(); rbs = scene['rbs'];
		ztr = -4.0
		if 1:
			rb = rb_create_box([3.0,1.5,0.2], opt_mass1); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['q'][2] = ztr; #rb['v'][0] = 0.5;
		if 1:
			rb = rb_create_box([1.0,1.0,1.0], 1.0); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['q'][2] = ztr; rb['q'][0] = 3.0/2.0;
		if ('1' in opt_ctr) and (len(rbs['bodies']) >= 2):
			sph = rb_spherical(rbs, 0, 1, ([3.0/2.0,0.0,ztr], 'w'), (v3_z(), 'l'))
			rbs_add_constraint(rbs, sph)
		if ('2' in opt_ctr) and (len(rbs['bodies']) >= 2):
			sph = rb_spherical(rbs, 0, 1, ([3.0/2.0,0.5,ztr], 'w'), (vec_add(rb_pt_world(rbs['bodies'][1], v3_z()), [0.0,0.5,0.0]) , 'w'))
			rbs_add_constraint(rbs, sph)
		if ('3' in opt_ctr and (not '1' in opt_ctr) and (not '2' in opt_ctr)) and (len(rbs['bodies']) >= 2):
			hng = rb_hinge(rbs, 0, 1, ([3.0/2.0,0.0,ztr], 'w'), ([0.0,0.5,0.0], 'w'), (v3_z(), 'l'), ([0.0,0.5,0.0], 'w'))
			rbs_add_constraint(rbs, hng)

	t = sctx['t']; dt = sctx['dt']
	rbs = scene['rbs']

	if (dt > 0.0):
		speed = float(arg_get('-speed', 1.0)); ampl = float(arg_get('-ampl', 1.0));
		if '1' in opt_wobble:
			rbs['bodies'][0]['v'][3+1] = ampl*0.8*math.sin(speed*2.0*t)
		if '2' in opt_wobble:
			rbs['bodies'][0]['v'][3+2] = ampl*0.2*1.0*math.sin(speed*0.8*t)
		if '3' in opt_wobble:
			rbs['bodies'][0]['v'][3+0] = ampl*0.2*1.0*math.sin(speed*0.8*t)
		rbs_step(rbs, 1.0/60.0, int(arg_get('-si_iters', 4)))
		return True

	return False

def scene_1_draw(sctx, col_func):
	scene = sctx['scene']
	rbs_draw(scene['rbs'], col_func)

test_k_shoulder = None
def scene_shoulder_input(*args):
	scene_def_input(*args)
	if args[0] == 'k':
		g_scene_context['scene']['kill'] = not (g_scene_context['scene']['kill'])
	if args[0] == 'u' and test_k_shoulder:
		rbs_remove_constraint(g_scene_context['scene']['rbs'], test_k_shoulder)
	if args[0] == 'i' and test_k_shoulder:
		rbs_add_constraint(g_scene_context['scene']['rbs'], test_k_shoulder)


def scene_shoulder_update(sctx):
	global test_k_shoulder
	scene = sctx['scene']
	opt_mode = str(arg_get('-mode', 'ik')) #ik,wobble,pendulum
	if (opt_mode == 'wobble'):
		opt_grav = float(arg_get('-grav', 0.0))
		opt_wobble = str(arg_get('-wobble', '2')).split(',')
		opt_mass1 = float(arg_get('-mass1', '2.6')) #or can be 'inf'
	else:
		opt_grav = float(arg_get('-grav', -10.0))
		opt_wobble = []
		opt_mass1 = 2.6
	opt_wobble_body = arg_get('-wobble_body', 'forearm')
	opt_exp_frames = int(arg_get('-exp_frames', 0))
	opt_exp_time = float(arg_get('-exp_time', 4.0))
	opt_exp_err_sq = float(arg_get('-exp_err_sq', 1.0e-2))
	opt_exp_sep = arg_get('-exp_sep', ' ')
	if (sctx['frame'] == 0):
		scene['kill'] = False
		scene['rbs'] = rbs_create(); rbs = scene['rbs']; rbs['g'] = [0.0,opt_grav,0.0];
		# http://www.exrx.net/Kinesiology/Segments.html
		if 1:
			rb = rb_create_box([1.0,1.0,1.0], float('inf')); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['name'] = 'shoulder'
		if 1:
			rb = rb_create_box([30.2,7.0,7.0], opt_mass1); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['q'][0] = (30.2/2.0);
			rb['name'] = 'upper-arm'
		if 1:
			rb = rb_create_box([26.9,5.0,5.0], 1.5); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['q'][0] = (30.2)+(26.9/2.0);
			rb['name'] = 'forearm'
		if True:
			b1,b2 = rbs_find_body_index(rbs, 'shoulder'), rbs_find_body_index(rbs, 'upper-arm')
			spt = (v3_z(), 'w')
			sph = rb_spherical(rbs, b1, b2, spt, spt)
			rbs_add_constraint(rbs, sph)
			test_k_shoulder = sph
		if 1:
			b1,b2 = rbs_find_body_index(rbs, 'upper-arm'), rbs_find_body_index(rbs, 'forearm')
			hpt = ([30.2,0.0,0.0], 'w'); haxis = ([0.0,0.0,4.0], 'w');
			hng = rb_hinge(rbs, b1, b2, hpt, haxis, hpt, haxis)
			rbs_add_constraint(rbs, hng)
			if arg_has('-_test3'):
				for rep in range(8):
					hng = rb_hinge(rbs, b1, b2, hpt, haxis, hpt, haxis)
					rbs_add_constraint(rbs, hng)
		if arg_has('-_test2'):
			b1,b2 = rbs_find_body_index(rbs, 'upper-arm'), rbs_find_body_index(rbs, 'forearm')
			hpt = ([15.2,0.0,0.0], 'w'); haxis = ([0.0,4.0,0.0], 'w');
			hng = rb_hinge(rbs, b1, b2, hpt, haxis, hpt, haxis)
			rbs_add_constraint(rbs, hng)
		if opt_mode == 'ik':
			rb = rb_create_box([1.0,1.0,1.0], float('inf')); obj_init_color(rb); rbs_add_body(rbs, rb);
			rb['q'][0] = (30.2)+(26.9);
			rb['name'] = 'ik'
			speed = float(arg_get('-speed', 1.0))
			anim_inclin = math.pi/2.0 if (not arg_has('-ik_3d')) else math.pi/4.0
			anim_state = rb_anim_sph_sample_func_state(7.0, 2.0*speed, mat_transl_R([(30.2)+(26.9/1.6),10.0,0.0], rv_to_mat([anim_inclin, 0.0, 0.0])))
			anim = rb_pt_anim(rbs, rbs_find_body_index(rbs, 'test'), rb_anim_sph_sample_func, anim_state)
			rbs_add_anim(rbs, anim)
		if opt_mode == 'ik':
			b1,b2 = rbs_find_body_index(rbs, 'forearm'), rbs_find_body_index(rbs, 'ik')
			spt = ([(30.2)+(26.9), 0.0, 0.0], 'w')
			sph = rb_spherical(rbs, b1, b2, spt, spt)
			rbs_add_constraint(rbs, sph)

		if (arg_has('-exp_file')):
			with open(arg_get('-exp_file', ''), 'w') as ofile:
				ofile.write('#header:{}\n'.format(arg_get('-exp_header', 'n/a')))
				ofile.write('#args:{}\n'.format(g_argv))
				ofile.write('#time:{}\n'.format(time.strftime("%d/%m/%Y-%H:%M:%S")))
				ofile.write('#sep:{}\n'.format(opt_exp_sep))
				exp_format = arg_get('-exp_format', 't,err_sq,1g1,2g2,g1,g2')
				ofile.write('#format:{}\n'.format(exp_format))
				scene['exp_fields'] = exp_format.split(',')
				scene['exp_lines'] = 0


	t = sctx['t']; dt = sctx['dt']; rbs = scene['rbs'];

	wobble_bi = rbs_find_body_index(rbs, opt_wobble_body)
	if (wobble_bi != -1):
		speed = float(arg_get('-speed', 1.0)); ampl = float(arg_get('-ampl', 1.0));
		ampl = (-ampl if opt_wobble_body == 'upper-arm' else ampl)
		if (not scene['kill']):
			if '1' in opt_wobble:
				rbs['bodies'][wobble_bi]['v'][3+1] = ampl*math.sin(speed*2.0*t)
			if '2' in opt_wobble:
				rbs['bodies'][wobble_bi]['v'][3+2] = ampl*math.sin(speed*2.0*t)
			if '3' in opt_wobble:
				rbs['bodies'][wobble_bi]['v'][3+0] = ampl*math.sin(speed*2.0*t)

	rbs_step(rbs, dt, int(arg_get('-si_iters', 8)))

	b1,b2 = rbs_find_bodies(rbs, ['upper-arm', 'forearm'])
	if ('exp_fields' in scene):
		if (opt_exp_err_sq == 0.0 or rbs['stats']['err_sq'] <= opt_exp_err_sq):
			with open(arg_get('-exp_file', ''), 'a') as ofile:
				for field in scene['exp_fields']:
					if field == 't':
						ofile.write('{}{}'.format(t, opt_exp_sep))
					elif field == 'err_sq':
						ofile.write('{}{}'.format(rbs['stats']['err_sq'], opt_exp_sep))
					elif field == 'lin_v1':
						for el in b1['v'][:3]:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					elif field == 'lin_v2':
						for el in b2['v'][:3]:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					elif field == '1g1':
						for el in b1['v'][3:]:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					elif field == 'g1':
						g1 = rb_vec_world(b1, b1['v'][3:])
						for el in g1:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					elif field == '2g2':
						for el in b2['v'][3:]:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					elif field == 'g2':
						g2 = rb_vec_world(b2, b2['v'][3:])
						for el in g2:
							ofile.write('{}{}'.format(el, opt_exp_sep))
					else:
						ofile.write('? ')
				ofile.write('\n')
				scene['exp_lines'] = scene['exp_lines']+1
		if (opt_exp_frames and sctx['frame'] > opt_exp_frames) or (opt_exp_time and sctx['t'] > opt_exp_time):
			scene_exit()
	if (arg_has('-print')):
		print_list = []
		if 'exp_lines' in scene:
			print_list.append('exported: {}'.format(scene['exp_lines']))
		if arg_has('-_test1'):
			th12 = (0.0,0.0)
			sth = [math.sin(x) for x in th12]; cth=[math.cos(x) for x in th12]
			def xfm_coord_simul(vp):
				return [-vp[2],vp[0],-vp[1]]
			def obj_p_i(w,i):
				return  (w[0]*sth[i]-w[2]*cth[i])**2 + w[1]**2
			def obj_d(w1,w2):
				return obj_p_i(w1,0) - obj_p_i(w2,1)
			dsamp = obj_d(xfm_coord_simul(b1['v'][3:]), xfm_coord_simul(b2['v'][3:]))
			print_list.append(vec_prec_str([dsamp],9))
		print_list.append('err_sq:{}, 1g1:{}, 2g2:{}'.format(rbs['stats']['err_sq'], vec_prec_str(b1['v'][3:], 3), vec_prec_str(b1['v'][3:], 3) ))
		if scene['kill']:
			print_list.append('kill')
		sctx['update_print'] = ', '.join(print_list)

	return True

def scene_shoulder_draw(sctx, col_func):
	scene = sctx['scene']
	rbs_draw(scene['rbs'], col_func)


def scene_chain_update(sctx):
	scene = sctx['scene']
	opt_len = int(arg_get('-length', 6))
	opt_grav = float(arg_get('-grav', -10.0))
	if (sctx['frame'] == 0):
		scene['rbs'] = rbs_create(); rbs = scene['rbs'];
		rbs['g'] = [0.0,opt_grav,0.0]
		ztr = 0.0
		prbi = -1
		for i in range(opt_len):
			rb = rb_create_box([1.0,1.0,1.0], float('inf') if (i == 0) else 1.0); obj_init_color(rb); rbi = rbs_add_body(rbs, rb);
			rb['q'][2] = ztr; rb['q'][1] = -float(i)*1.0; rb['q'][0] = -(0.5+float(i-1)) if i>0 else 0.0;
			if (i > 0):
				pvt = vec_add(rb['q'][:3], [0.5,0.5,0.0])
				sph = rb_spherical(rbs, prbi, rbi, (pvt, 'w'), (pvt, 'w'))
				rbs_add_constraint(rbs, sph)
			prbi = rbi

	rbs_step(scene['rbs'], sctx['dt'], int(arg_get('-si_iters', 8)))

	return True

def scene_chain_draw(sctx, col_func):
	scene = sctx['scene']
	rbs_draw(scene['rbs'], col_func)


################################################################################
# Scene Drawing
################################################################################
col_wt = [1.0,1.0,1.0]
col_rd = [1.0,0.0,0.0]
col_grn = [0.0,1.0,0.0]
col_bl = [0.0,0.0,1.0]
col_blk = [0.0,0.0,0.0]

def hsv2rgb(h, s, v):
	hi = int(h*6)
	f = h*6 - hi
	p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s);
	r, g, b = 0, 0, 0
	if hi == 0: r, g, b = v, t, p
	elif hi == 1: r, g, b = q, v, p
	elif hi == 2: r, g, b = p, v, t
	elif hi == 3: r, g, b = p, q, v
	elif hi == 4: r, g, b = t, p, v
	elif hi == 5: r, g, b = v, p, q
	return [r, g, b]

g_randcol_h = 0.0
def randcol():
	global g_randcol_h
	golden_ratio_conjugate = 0.618033988749895
	g_randcol_h = g_randcol_h + golden_ratio_conjugate
	g_randcol_h = g_randcol_h % 1.0
	return hsv2rgb(g_randcol_h, 0.5, 0.95)

def obj_init_color(obj):
	if arg_has('-mono'):
		obj['col'] = col_wt
	else:
		obj['col'] = randcol()

def scene_empty_update(sctx):
	return True

def scene_empty_draw(sctx, col_func):
	return

g_scene_context = {
	'wind_handle' : 0,
	'wind_w' : 640,
	'wind_h' : 480,
	'wind_viz' : None,
	'update_func' : scene_empty_update,
	'draw_func' : scene_empty_draw,
	'frame' : 0, 't' : 0.0, 'dt' : 0.0, 'last_clock_mt' : 0,
	'fixed_dt' : 0, 'adapt_fixed_dt': True,
	'paused' : False, 'paused_step' : False,
	'loop_frame' : 0,
	'fps_last_clock_mt': 0, 'fps_marker' : 0, 'fps' : 0,
	'scene': {},
	'update_print': '',
	'ViewM': mat_id(), 'CamM': mat_id(),
	'cam_coords': [0.0, math.pi/2.0, 30.0], 'cam_def_coords': [0.0, math.pi/2.0, 30.0], 'cam_speeds':[0.06, 0.06, 0.6],
	'cam_keys': {'w':[0.0,0.0,-1.0], 's':[0.0,0.0,1.0], 'a':[-1.0,0.0,0.0], 'd':[1.0,0.0,0.0], 'q':[0.0,-1.0,0.0], 'e':[0.0,1.0,0.0]},
	'cam_key_on': {}
	}

def scene_viewport_init(w, h, aa = True):
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glCullFace(GL_BACK)
	glClearDepth(1.0)
	glDepthFunc(GL_LESS)
	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(60.0, float(w)/float(h), 0.01, 1000.0)
	glMatrixMode(GL_MODELVIEW)
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )

	glEnable(GL_MULTISAMPLE)
	if (aa):
		glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable (GL_BLEND)
		glEnable (GL_LINE_SMOOTH)
		glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)
		glEnable (GL_POLYGON_SMOOTH)
		glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST)

def scene_exit():
	if arg_has(['-print']):
		print '\n'
	sys.exit()

def scene_update_cam():
	ctx = g_scene_context; crds = ctx['cam_coords'];
	for k in ctx['cam_key_on']:
		if (ctx['cam_key_on'][k]):
			vec = ctx['cam_keys'][k]
			ci = [abs(x) for x in vec].index(1.0)
			vec = vec_muls(vec, ctx['cam_speeds'][ci])
			ctx['cam_coords'] = vec_add(ctx['cam_coords'], vec)
	ctx = g_scene_context; crds = ctx['cam_coords'];
	crds[2] = max(0.001, crds[2])
	crds[1] = min(math.pi-0.001, max(0.001, crds[1]))
	pt = coord_sph_to_cart2(crds);
	cz = vec_normd(pt); cx = v3_cross([0.0,1.0,0.0], cz);
	if (vec_norm(cx) <= 1.e-6):
		cx = v3_cross(mat_transp(ctx['CamM'])[1].tolist(), cz)
	cy = vec_normd(v3_cross(cz, cx)); cx = vec_normd(v3_cross(cy, cz));
	mat_set_transl(ctx['CamM'], pt); mat_set_rot(ctx['CamM'], mat_transp([ cx, cy, cz ]));
	g_scene_context['ViewM'] = mat_rigid_inv(g_scene_context['CamM'])

def scene_def_input(*args):
	ctx = g_scene_context
	if args[0] == '\033':
		scene_exit()
	elif args[0] == '\r':
		ctx['paused'] = not ctx['paused']
	elif args[0] == ' ':
		ctx['paused_step'] = True
	elif (args[0] in ctx['cam_keys']):
		ctx['cam_key_on'][args[0]] = True
	elif (args[0] == 'r'):
		ctx['cam_coords'][:2] = ctx['cam_def_coords'][:2]
	#else:
	#    print args[0]

def scene_def_up_input(*args):
	ctx = g_scene_context
	if (args[0] in ctx['cam_keys']):
		ctx['cam_key_on'][args[0]] = False

def scene_color_zero(r,g,b):
	glColor3f(0.0,0.0,0.0)

def scene_color_pass(r,g,b):
	glColor3f(r,g,b)

def scene_visibility_func(state):
	g_scene_context['wind_vis'] = state

def scene_idle_func():
	if (g_scene_context['wind_vis'] == GLUT_VISIBLE):
		scene_loop_func()
	return


def handle_console_input():
	if fcntl is not None:
		ctx = g_scene_context
		if (ctx['loop_frame'] == 0):
			fd = sys.stdin.fileno()
			fl = fcntl.fcntl(fd, fcntl.F_GETFL)
			fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
		try:
			input = sys.stdin.readline()
			print 'echo:', input,
		except:
			return


def scene_loop_func():
	handle_console_input()
	try:
		print_arg_help()
		ctx = g_scene_context
		scene_update_cam()
		if (ctx['loop_frame'] == 0):
			ctx['last_clock_mt'] = glutGet(GLUT_ELAPSED_TIME)
		clock_mt = glutGet(GLUT_ELAPSED_TIME)
		mdt = clock_mt - ctx['last_clock_mt']
		if (mdt > 0):

			ctx['last_clock_mt'] = clock_mt

			do_draw = True; do_frame = False;
			if (ctx['paused'] == False) or (ctx['paused_step'] == True):
				if (ctx['fixed_dt'] != 0):
					dt_count = 1
					if (ctx['adapt_fixed_dt'] and (not ctx['paused_step'])):
						dt_count =  max(1, min(8, int(math.ceil((mdt/1000.0) / ctx['fixed_dt']))))
					for i in range(dt_count):
						ctx['dt'] = ctx['fixed_dt']
						do_frame = ctx['update_func'](ctx)
						if (do_frame):
							ctx['t'] = ctx['t'] + ctx['fixed_dt']
				else:
					ctx['dt'] = mdt/1000.0;
					do_frame = ctx['update_func'](ctx)
					if (do_frame):
						ctx['t'] = ctx['t'] + mdt/1000.0;
			else:
				if (ctx['frame'] == 0):
					do_draw = False
			ctx['paused_step'] = False

			if arg_has(['-print']) or len(ctx['update_print']):
				if (ctx['loop_frame'] > 0):
					sys.stdout.write('\x1B[2K')

				print_strs = []
				if arg_has(['-print']):
					if (clock_mt - ctx['fps_last_clock_mt'] >= 1000):
						ctx['fps'] = (ctx['loop_frame'] - ctx['fps_marker']) / ((clock_mt - ctx['fps_last_clock_mt'])/1000.0)
						ctx['fps_last_clock_mt'] = clock_mt; ctx['fps_marker'] = ctx['loop_frame'];
					print_strs.append('frame: {}, time: {:.3f}, fps: {:.2f}'.format(ctx['frame'], ctx['t'], ctx['fps']))
				if len(ctx['update_print']):
					print_strs.append(ctx['update_print'])
				sys.stdout.write('\r{}'.format(' '.join(['[{}]'.format(x) for x in print_strs]) ))
				sys.stdout.flush()

			if (do_draw):
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

				if (arg_has('-fill')):
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					glMatrixMode(GL_MODELVIEW)
					ctx['draw_func'](ctx, scene_color_pass)
				else:
					if (not arg_has('-fancy')):
						glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
						glMatrixMode(GL_MODELVIEW)
						ctx['draw_func'](ctx, scene_color_pass)
					else:
						glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
						glMatrixMode(GL_MODELVIEW)
						glColor3f(0.0,0.0,0.0)
						ctx['draw_func'](ctx, scene_color_zero)

						glEnable(GL_POLYGON_OFFSET_LINE);
						glPolygonOffset(-1,-1);

						glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
						glMatrixMode(GL_MODELVIEW)
						ctx['draw_func'](ctx, scene_color_pass)

						glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
						glDisable(GL_POLYGON_OFFSET_LINE);

				glutSwapBuffers()

			if (do_frame):
				ctx['frame'] = ctx['frame']+1

		ctx['loop_frame'] = ctx['loop_frame']+1
	except:
		traceback.print_exc()
		sys.exit()

def scene_xfm(M):
	glLoadMatrixf(mat_transp(mat_mul(g_scene_context['ViewM'], M)))

def scene_xfm_id():
	glLoadMatrixf(mat_transp(g_scene_context['ViewM']))

def scene_draw_cs(basis, col_func):
	idb = [(1.0,0,0), (0,1.0,0), (0,0,1.0)]
	for b,ax in zip(basis, idb):
		scene_draw_line(v3_z(), [b*x for x in ax], ax, col_func)

def scene_draw_line(v1, v2, col, col_func):
	col_func(*col); glBegin(GL_LINES); glVertex3f(*v1); glVertex3f(*v2); glEnd();

def scene_draw_box(dim=[1.0,1.0,1.0]):
	dim = [0.5*x for x in dim]
	glBegin(GL_QUADS)
	glVertex3f(dim[0], dim[1],-dim[2]); glVertex3f(-dim[0], dim[1],-dim[2]); glVertex3f(-dim[0], dim[1], dim[2]); glVertex3f(dim[0], dim[1], dim[2]);
	glVertex3f(dim[0],-dim[1], dim[2]); glVertex3f(-dim[0],-dim[1], dim[2]); glVertex3f(-dim[0],-dim[1],-dim[2]); glVertex3f(dim[0],-dim[1],-dim[2]);
	glVertex3f(dim[0], dim[1], dim[2]); glVertex3f(-dim[0], dim[1], dim[2]); glVertex3f(-dim[0],-dim[1], dim[2]); glVertex3f(dim[0],-dim[1], dim[2]);
	glVertex3f(dim[0],-dim[1],-dim[2]); glVertex3f(-dim[0],-dim[1],-dim[2]); glVertex3f(-dim[0], dim[1],-dim[2]); glVertex3f(dim[0], dim[1],-dim[2]);
	glVertex3f(-dim[0], dim[1], dim[2]); glVertex3f(-dim[0], dim[1],-dim[2]); glVertex3f(-dim[0],-dim[1],-dim[2]); glVertex3f(-dim[0],-dim[1], dim[2]);
	glVertex3f(dim[0], dim[1],-dim[2]); glVertex3f(dim[0], dim[1], dim[2]); glVertex3f(dim[0],-dim[1], dim[2]); glVertex3f(dim[0],-dim[1],-dim[2]);
	glEnd()

def scene_draw_point(dim=1.0):
	glBegin(GL_POINTS)
	glVertex3f(*v3_z());
	glEnd();

def scene_test_draw(sctx, col_func):
	t = sctx['t']

	glLoadMatrixf(mat_transp(mat_transl_aa((0.0,0.0,-18.0), [1.0,1.0,0.0,t])))
	col_func(1.0,1.0,1.0)
	scene_draw_box([2.0,1.0,0.5])

	glLoadMatrixf(mat_transp(mat_transl_aa((-3.0,0.0,-12.0), [1.0,0.0,1.0,t*0.5])))
	col_func(0.0,0.0,1.0)
	scene_draw_box()

#http://stackoverflow.com/questions/1892339/how-to-make-a-window-jump-to-the-front
import subprocess
def pyogl_mac_focus_hack():
	def applescript(script):
		return subprocess.check_output(['/usr/bin/osascript', '-e', script])
	applescript('''
		tell app "System Events"
			repeat with proc in every process whose name is "Python"
				set frontmost of proc to true
				exit repeat
			end repeat
		end tell''')

def scene_go(title, update_func, draw_func, input_func = scene_def_input):
	global g_scene_context

	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | (GLUT_MULTISAMPLE if arg_has('-multisample') else 0) ) #
	glutInitWindowSize(g_scene_context['wind_w'], g_scene_context['wind_h'])
	glutInitWindowPosition(200,200)

	g_scene_context['wind_handle'] = glutCreateWindow(title)
	g_scene_context['update_func'] = update_func
	g_scene_context['draw_func'] = draw_func
	g_scene_context['paused'] = arg_has('-paused')
	if (not arg_has_key('-flex_dt')):
		fixed_dt = float(eval(arg_get('-dt', '1.0/60.0')))
		if (fixed_dt <= 0):
			fixed_dt = float(eval(arg_get('-dt','')+'.0'))
		if (fixed_dt <= 0):
			sys.exit()
		g_scene_context['fixed_dt'] = fixed_dt
		g_scene_context['adapt_fixed_dt'] = arg_has('-adapt_fixed_dt')

	glutDisplayFunc(scene_loop_func)
	glutIdleFunc(scene_idle_func)
	glutVisibilityFunc(scene_visibility_func)
	glutKeyboardFunc(input_func)
	glutKeyboardUpFunc(scene_def_up_input)
	scene_viewport_init(g_scene_context['wind_w'], g_scene_context['wind_h'])
	if True:
		pyogl_mac_focus_hack()
	glutMainLoop()

################################################################################
# More scenes
################################################################################

def scene_rbcable1_update(sctx):
	scene = sctx['scene']
	opt_len = int(arg_get('-length', 10))
	opt_grav = float(arg_get('-grav', -10.0))
	opt_load = float(arg_get('-load', 200.0))
	opt_pert = float(arg_get('-pert', 0.3))
	opt_iters = int(arg_get('-si_iters', 12))
	if (sctx['frame'] == 0):
		scene['rbs'] = rbs_create(); rbs = scene['rbs'];
		rbs['g'] = [0.0,opt_grav,0.0]
		ztr = 0.0
		prbi = -1
		el_h = 1.0; el_w = 0.1;
		for i in range(opt_len):
			rb = rb_create_box([el_w,el_h,el_w], float('inf') if (i == 0) else 1.0); obj_init_color(rb); rbi = rbs_add_body(rbs, rb);
			obj_init_color(rb)
			rb['q'][2] = ztr; rb['q'][1] = -float(i)*el_h; rb['q'][0] = 0.0;
			if (i > 0):
				pvt = vec_add(rb['q'][:3], [0.0,el_h*0.5,0.0])
				sph = rb_spherical(rbs, prbi, rbi, (pvt, 'w'), (pvt, 'w'))
				rbs_add_constraint(rbs, sph)
				if opt_pert > 0.0 and prbi >= 1:
					prbq = rbs_get_body(rbs, prbi)['q']
					prbq[3:] = [prbq[3:][i] + rand_ampl(opt_pert) for i in range(4)]
					prbq[3:] = vec_normd(prbq[3:])
			prbi = rbi
		if opt_load > 0.0:
			rb = rb_create_box([2.0*el_w,2.0*el_w,2.0*el_w], opt_load); obj_init_color(rb); rbi = rbs_add_body(rbs, rb);
			obj_init_color(rb)
			rb['q'][2] = ztr; rb['q'][1] = -(float(opt_len-1)*el_h+0.5*el_h); rb['q'][0] = 0.0;
			if (prbi >= 0):
				pvt = vec_add(rb['q'][:3], [0.0,el_h*0.5,0.0])
				sph = rb_spherical(rbs, prbi, rbi, (pvt, 'w'), (pvt, 'w'))
				rbs_add_constraint(rbs, sph)
			prbi = rbi

	rbs_step(scene['rbs'], sctx['dt'], opt_iters)

	return True

def scene_pendulum_update(sctx):
	scene = sctx['scene']
	opt_n = int(arg_get('-n', 1))
	opt_len = int(arg_get('-length', 3.0))
	opt_grav = float(arg_get('-grav', -10.0))
	opt_load = float(arg_get('-load', 1.0))
	opt_pert = float(arg_get('-pert', 0.0))
	opt_iters = int(arg_get('-si_iters', 1))
	if (sctx['frame'] == 0):
		scene['rbs'] = rbs_create(); rbs = scene['rbs'];
		rbs['g'] = [0.0,opt_grav,0.0]
		ztr = 0.0
		prbi = -1
		el_h = opt_len; el_w = 0.1;
		for i in range(opt_n + 1):
			rb = rb_create_box([el_w,el_w if (i==0) else el_h,el_w], float('inf') if (i == 0) else opt_load); obj_init_color(rb); rbi = rbs_add_body(rbs, rb);
			obj_init_color(rb)
			rb['q'][2] = ztr;
			if (i > 0):
				rb['q'][1] = 0.0; rb['q'][0] = 0.0*el_w + 0.5*el_h + float(i-1)*el_h;
				rb['q'][3:] = rv_to_uquat([0.0, 0.0, deg_rad(90.0)])
				pvt = vec_add(rb['q'][:3], [-el_h*0.5,0.0,0.0])
				sph = rb_spherical(rbs, prbi, rbi, (pvt, 'w'), (pvt, 'w'))
				rbs_add_constraint(rbs, sph)
				if opt_pert > 0.0 and prbi >= 1:
					prbq = rbs_get_body(rbs, prbi)['q']
					prbq[3:] = [prbq[3:][i] + rand_ampl(opt_pert) for i in range(4)]
					prbq[3:] = vec_normd(prbq[3:])
			prbi = rbi
	rbs_step(scene['rbs'], sctx['dt'], opt_iters)
	return True

def scene_default_draw(sctx, col_func):
	scene = sctx['scene']
	rbs_draw(scene['rbs'], col_func)

if __name__ == "__main__":
	scene_name = arg_get('-scene', 'test')
	scene_title = 'Scene: {}'.format(scene_name)
	if (scene_name == '1'):
		scene_go(scene_title, scene_1_update, scene_1_draw)
	elif (scene_name == 'shoulder'):
		scene_go(scene_title, scene_shoulder_update, scene_shoulder_draw, scene_shoulder_input)
	elif (scene_name == 'chain'):
		scene_go(scene_title, scene_chain_update, scene_chain_draw)
	elif (scene_name == 'cable'):
		scene_go(scene_title, scene_rbcable1_update, scene_default_draw)
	elif (scene_name == 'pendulum'):
		scene_go(scene_title, scene_pendulum_update, scene_default_draw)
	else:
		scene_go(scene_title, scene_empty_update, scene_test_draw)
