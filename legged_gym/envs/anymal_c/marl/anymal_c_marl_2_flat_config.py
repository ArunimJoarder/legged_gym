# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs import AnymalCFlatCfg, AnymalCFlatCfgPPO

class AnymalCMARL_2_FlatCfg( AnymalCFlatCfg ):
	class env( AnymalCFlatCfg.env ):
		num_agents = 2
		num_dof_per_agent = 12//num_agents
		num_actions_per_agent = 12//num_agents
		num_obs_per_agent = 48
		num_privileged_obs_per_agent = 48

	class asset( AnymalCFlatCfg.asset ):
		penalize_contacts_on = ["LF_SHANK", "LF_THIGH", "LH_SHANK", "LH_THIGH", "RF_SHANK", "RF_THIGH", "RH_SHANK", "RH_THIGH"]
		pass

	class rewards( AnymalCFlatCfg.rewards ):
		class scales( AnymalCFlatCfg.rewards.scales ):
			# pass
			# termination = -0.0
			# tracking_lin_vel = 1.0
			# tracking_ang_vel = 0.5
			# lin_vel_z = -2.0
			# ang_vel_xy = -0.05
			# dof_vel = -0.
			# dof_acc = -2.5e-7
			# base_height = -0. 
			# collision = -1.
			# feet_stumble = -0.0 
			# action_rate = -0.01
			# stand_still = -0.
			# orientation = -5.0
			# torques = -0.000025
			# feet_air_time = 2.
			
			dof_acc = 0.
			dof_vel = 0.
			feet_air_time =  0.
			action_rate = 0.
			torques = 0.
			
			orientation = -5.0
		
			termination = -0.0
			tracking_lin_vel = 1.0
			tracking_ang_vel = 0.5
			lin_vel_z = -2.0
			ang_vel_xy = -0.05
			base_height = 0. 
			collision = 0.
			feet_stumble = -0.0 
			stand_still = -0.

			agent_1_2_torques = -0.000025
			agent_1_2_dof_acc = -2.5e-7
			agent_1_2_action_rate = -0.01
			agent_1_2_foot_air_time =  2.
			agent_1_2_collision = -1.

			agent_2_2_torques = -0.000025
			agent_2_2_dof_acc = -2.5e-7
			agent_2_2_action_rate = -0.01
			agent_2_2_foot_air_time =  2.
			agent_2_2_collision = -1.

class AnymalCMARL_2_FlatCfgPPO( AnymalCFlatCfgPPO ):
	class policy( AnymalCFlatCfgPPO.policy ):
		actor_hidden_dims = [64, 64, 16]
		critic_hidden_dims = [64, 64, 16]
		activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
		pass

	class runner ( AnymalCFlatCfgPPO.runner):
		experiment_name = 'marl_2_flat_anymal_c'
		max_iterations = 300 # number of policy updates
