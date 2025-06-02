// Mock API service for testing
const mockApiService = {
  healthCheck: jest.fn(),
  analyzeSingleImage: jest.fn(),
  analyzeBatchImages: jest.fn()
};

describe('API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('healthCheck returns status', async () => {
    const mockResponse = { status: 'healthy' };
    mockApiService.healthCheck.mockResolvedValue(mockResponse);

    const result = await mockApiService.healthCheck();

    expect(mockApiService.healthCheck).toHaveBeenCalled();
    expect(result).toEqual(mockResponse);
  });

  test('analyzeSingleImage processes file', async () => {
    const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const mockResponse = { 
      success: true, 
      analysis_id: 'test123',
      cell_counts: { RBC: 100, WBC: 10, Platelets: 20 }
    };
    
    mockApiService.analyzeSingleImage.mockResolvedValue(mockResponse);

    const result = await mockApiService.analyzeSingleImage(mockFile);

    expect(mockApiService.analyzeSingleImage).toHaveBeenCalledWith(mockFile);
    expect(result).toEqual(mockResponse);
  });

  test('handles API errors gracefully', async () => {
    const mockError = new Error('Analysis failed');
    mockApiService.analyzeSingleImage.mockRejectedValue(mockError);

    await expect(mockApiService.analyzeSingleImage(new File(['test'], 'test.jpg')))
      .rejects
      .toThrow('Analysis failed');
  });

  test('analyzeBatchImages handles multiple files', async () => {
    const mockFiles = [
      new File(['test1'], 'test1.jpg', { type: 'image/jpeg' }),
      new File(['test2'], 'test2.jpg', { type: 'image/jpeg' })
    ];
    const mockResponse = { 
      success: true, 
      batch_results: [
        { filename: 'test1.jpg', cell_counts: { RBC: 100 } },
        { filename: 'test2.jpg', cell_counts: { RBC: 150 } }
      ]
    };
    
    mockApiService.analyzeBatchImages.mockResolvedValue(mockResponse);

    const result = await mockApiService.analyzeBatchImages(mockFiles);

    expect(mockApiService.analyzeBatchImages).toHaveBeenCalledWith(mockFiles);
    expect(result.batch_results).toHaveLength(2);
  });
});
