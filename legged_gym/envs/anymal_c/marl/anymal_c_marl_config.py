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

class AnymalCMARLCfg( AnymalCFlatCfg ):
	class env( AnymalCFlatCfg.env ):
		num_agents = 4

	class rewards( AnymalCFlatCfg.rewards ):
		class scales( AnymalCFlatCfg.rewards.scales ):
			termination = -0.0
			tracking_lin_vel = 1.0
			tracking_ang_vel = 0.5
			lin_vel_z = -2.0
			ang_vel_xy = -0.05
			orientation = -5.0
			dof_vel = 0.
			dof_acc = 0.
			base_height = -0. 
			feet_air_time =  0.
			collision = 0.
			feet_stumble = 0.
			action_rate = 0.
			stand_still = 0.
			
			leg_1_torques = -0.000025
			leg_1_dof_acc = -2.5e-7
			leg_1_feet_air_time = 2.
			leg_1_collision = -1.
			leg_1_action_rate = -0.01

			leg_2_torques = -0.000025
			leg_2_dof_acc = -2.5e-7
			leg_2_feet_air_time = 2.
			leg_2_collision = -1.
			leg_2_action_rate = -0.01

			leg_3_torques = -0.000025
			leg_3_dof_acc = -2.5e-7
			leg_3_feet_air_time = 2.
			leg_3_collision = -1.
			leg_3_action_rate = -0.01

			leg_4_torques = -0.000025
			leg_4_dof_acc = -2.5e-7
			leg_4_feet_air_time = 2.
			leg_4_collision = -1.
			leg_4_action_rate = -0.01


class AnymalCMARLCfgPPO( AnymalCFlatCfgPPO ):
    class runner ( AnymalCFlatCfgPPO.runner):
        experiment_name = 'marl_anymal_c'
