/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*!
 * \file ifa_flag_data.h
 * \brief
 */

#ifndef IFA_FLAG_DATA_H
#define IFA_FLAG_DATA_H

struct IFAFlagData {
    uint64_t tscmIdx : 2; // query tscm que idx
    uint64_t tscmReuse : 1;
    uint64_t rsvd : 61; // 保留
};

#endif // IFA_FLAG_DATA_H
