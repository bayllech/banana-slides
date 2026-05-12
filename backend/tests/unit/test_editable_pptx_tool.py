import io
from unittest.mock import patch

from PIL import Image


def _png_bytes(color='red', size=(320, 180)):
    buffer = io.BytesIO()
    Image.new('RGB', size, color=color).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


def test_editable_pptx_tool_requires_files(client):
    response = client.post('/api/projects/editable-pptx', data={}, content_type='multipart/form-data')

    assert response.status_code == 400
    data = response.get_json()
    assert data.get('success') is False


@patch('controllers.project_controller.task_manager.submit_task')
def test_editable_pptx_tool_accepts_multiple_images(mock_submit_task, client, app):
    response = client.post(
        '/api/projects/editable-pptx',
        data={
            'files': [
                (_png_bytes('red'), 'slide-1.png'),
                (_png_bytes('blue'), 'slide-2.png'),
            ],
            'filename': 'converted.pptx',
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 202
    data = response.get_json()
    assert data.get('success') is True
    payload = data['data']
    assert payload['page_count'] == 2
    assert payload['filename'] == 'converted.pptx'

    with app.app_context():
        from models import Page, Project, Task

        project = Project.query.get(payload['project_id'])
        assert project is not None
        assert project.creation_type == 'editable_import'
        assert project.image_aspect_ratio == '16:9'
        assert project.export_allow_partial is False
        assert project.export_inpaint_method == 'hybrid'
        assert project.enable_icon_subject_extraction is True

        pages = Page.query.filter_by(project_id=payload['project_id']).order_by(Page.order_index).all()
        assert len(pages) == 2
        assert all(page.generated_image_path for page in pages)

        task = Task.query.get(payload['task_id'])
        assert task is not None
        assert task.task_type == 'EXPORT_EDITABLE_PPTX'
        assert task.status == 'PENDING'

    mock_submit_task.assert_called_once()
    assert mock_submit_task.call_args.kwargs['project_id'] == payload['project_id']
    assert mock_submit_task.call_args.kwargs['filename'] == 'converted.pptx'
    assert mock_submit_task.call_args.kwargs['export_inpaint_method'] == 'hybrid'
    assert mock_submit_task.call_args.kwargs['enable_icon_subject_extraction'] is True
    assert mock_submit_task.call_args.kwargs['extract_text_styles'] is True


@patch('controllers.project_controller.task_manager.submit_task')
def test_editable_pptx_tool_can_disable_text_styles(mock_submit_task, client):
    response = client.post(
        '/api/projects/editable-pptx',
        data={
            'files': [(_png_bytes('red'), 'slide-1.png')],
            'extract_text_styles': 'false',
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 202
    assert mock_submit_task.call_args.kwargs['extract_text_styles'] is False


def test_editable_pptx_tool_rejects_unsupported_file(client):
    response = client.post(
        '/api/projects/editable-pptx',
        data={
            'files': [(io.BytesIO(b'hello'), 'notes.txt')],
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    data = response.get_json()
    assert data.get('success') is False
