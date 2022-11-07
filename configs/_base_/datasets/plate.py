dataset_info = dict(
    dataset_name='plate-kpts',
    paper_info=dict(
        author='Hieu Trinh',
        title='Detect 4 corner points of plate',
        container='',
        year='2022',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='1-TL', 
            id=0, 
            color=[255, 0, 0], 
            type='', 
            swap='2-TR'),
        1:
        dict(
            name='2-TR',
            id=1,
            color=[0, 255, 0],
            type='',
            swap='1-TL'),
        2:
        dict(
            name='3-BR',
            id=2,
            color=[0, 0, 255],
            type='',
            swap='4-BL'),
        3:
        dict(
            name='4-BL',
            id=3,
            color=[255, 255, 0],
            type='',
            swap='3-BR'),
    },
    skeleton_info={
        0:
        dict(link=('1-TL', '2-TR'), id=0, color=[0, 128, 255]),
        1:
        dict(link=('2-TR', '3-BR'), id=1, color=[0, 255, 128]),
        2:
        dict(link=('3-BR', '4-BL'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('4-BL', '1-TL'), id=3, color=[255, 0, 128]),
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.1, 0.1, 0.1, 0.1
    ])
